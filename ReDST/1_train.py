import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from helper import MultiwozDSTData
from model import RNNDST, MLPDST


def train(conf):
    dataset = MultiwozDSTData(conf)
    conf["num_label_list"] = dataset.num_label_list 
    conf["num_slots"] = len(dataset.slot_id_map)
    conf["num_labels"] = len(dataset.label_id_map)
    print("number of slots: ", conf["num_slots"])
    print("number of labels for each slot: ")
    for slot, i in sorted(dataset.slot_id_map.items(), key=lambda i: i[1]):
        print("%s : %s" %(slot, len(dataset.slot_label_id_map[slot])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    conf["device"] = device

    if conf["dst_model"] == "RNN":
        model = RNNDST(conf, [dataset.adj, dataset.ent2attr_adj, dataset.attr2ent_adj])
    if conf["dst_model"] == "MLP":
        model = MLPDST(conf, dataset.adj)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"])

    best_joint_acc, best_joint_acc_epoch, best_slot_acc, best_slot_acc_epoch = 0, 0, 0, 0
    for epoch in range(conf["epoch"]):
        print("\nEpoch: %d" %(epoch))
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        loss_print, ori_loss_print, each_acc_print = 0, 0, 0
        epoch_num = torch.FloatTensor([epoch]).to(device)
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()

            [last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, _] = [each.to(device) for each in batch]
            losses, _ = model(last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, epoch=epoch_num)
            ori_loss = sum(losses) / len(losses)
            each_acc = 0
            loss = ori_loss

            loss.backward()
            optimizer.step()

            loss_print += loss.item()
            ori_loss_print += ori_loss.item()
            each_acc_print += each_acc
            pbar.set_description('L:{:.4f} OL:{:.4f} B_acc:{:.4f}'.format(loss_print/(batch_i+1), ori_loss_print/(batch_i+1), each_acc_print/(batch_i+1)))

        if (epoch+1) % 1 == 0:
            #print("Evaluate on dev set:")
            #evaluate(model, dataset, device, which="dev")
            print("Evaluate on test set:")
            joint_acc, slot_acc = evaluate(model, dataset, device, best_joint_acc, conf, which="test")
            if joint_acc > best_joint_acc:
                best_joint_acc = joint_acc
                best_joint_acc_epoch = epoch
            if slot_acc > best_slot_acc:
                best_slot_acc = slot_acc
                best_slot_acc_epoch = epoch
            print("Best Joint ACC (epoch %d): %f; Best Slot ACC (epoch %d): %f" 
                      %(best_joint_acc_epoch, best_joint_acc, best_slot_acc_epoch, best_slot_acc))

        id_pred_map_train = pred_train_next(model, dataset, device, conf, which="train")
        id_pred_map_dev = pred_train_next(model, dataset, device, conf, which="dev")
        id_pred_map_test = pred_train_next(model, dataset, device, conf, which="test")
        # use the id_pred_map update the dataset
        dataset.init_dataset(train_id_pred_map=id_pred_map_train, dev_id_pred_map=id_pred_map_dev, test_id_pred_map=id_pred_map_test)

    return dataset


def evaluate_one_batch(model, batch, device, dataset):
    model.eval()
    [last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief] = [each.to(device) for each in batch]
    losses, preds = model(last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief)

    all_grd = current_belief.detach().cpu().numpy().tolist()
    each_pred = []
    for x in preds:
        x = x.detach().cpu() # [batch_size, label_num]
        x_idx = torch.argmax(x, dim=-1).numpy().tolist() # [batch_size] 
        each_pred.append(x_idx) 
    # each_pred: [num_slots, batch_size]
    each_pred = np.array(each_pred).transpose(1, 0).tolist()
    all_pred = each_pred

    slot_id_label_map, id_slot_map = dataset.slot_id_label_map, dataset.id_slot_map
    extracted_grd, extracted_pred = [], []
    for grd, pred in zip(all_grd, all_pred):
        grd_set, pred_set = set(), set()
        for i, slot in sorted(id_slot_map.items(), key=lambda i: int(i[0])):
            i = int(i)
            grd_str = slot_id_label_map[slot][str(grd[i])]
            if grd_str != "none":
                grd_set.add(grd_str)
            pred_str = slot_id_label_map[slot][str(pred[i])]
            if pred_str != "none":
                pred_set.add(pred_str)
        extracted_grd.append(grd_set)
        extracted_pred.append(pred_set)
    acc = cal_metric_for_each(extracted_grd, extracted_pred)
    model.train(True)
    return acc


def evaluate(model, dataset, device, best_joint_acc, conf, which="dev"):
    bad_set = set(dataset.bad_set[which])
    id_data_map = dataset.id_data_map[which]
    if which == "dev":
        the_loader = dataset.dev_loader
    if which == "test":
        the_loader = dataset.test_loader
    model.eval()
    pbar = tqdm(enumerate(the_loader), total=len(the_loader))
    all_grd, all_pred, all_data_idx, cont_ids = [], [], [], []
    loss_print = 0
    for batch_i, batch in pbar:
        [last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, data_idx] = [each.to(device) for each in batch]
        losses, preds = model(last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief)

        loss = sum(losses) / len(losses) 
        loss_print += loss.item()
        pbar.set_description('L:{:.4f}'.format(loss_print/(batch_i+1)))

        all_grd += current_belief.detach().cpu().numpy().tolist()
        all_data_idx += data_idx.squeeze(1).cpu().numpy().tolist()
        each_pred = []
        for x in preds:
            x = x.detach().cpu() # [batch_size, label_num]
            x_idx = torch.argmax(x, dim=-1).numpy().tolist() # [batch_size] 
            each_pred.append(x_idx) 
        # each_pred: [num_slots, batch_size]
        each_pred = np.array(each_pred).transpose(1, 0).tolist()
        all_pred += each_pred

    slot_id_label_map, id_slot_map = dataset.slot_id_label_map, dataset.id_slot_map
    extracted_grd, extracted_pred, data_ids = [], [], []
    for grd, pred, idx in zip(all_grd, all_pred, all_data_idx):
        grd_set, pred_set = set(), set()
        for i, slot in sorted(id_slot_map.items(), key=lambda i: int(i[0])):
            i = int(i)
            grd_str = slot_id_label_map[slot][str(grd[i])]
            if grd_str != "none":
                grd_str = "-".join([slot, grd_str])
                grd_set.add(grd_str)
            pred_str = slot_id_label_map[slot][str(pred[i])]
            if pred_str != "none":
                pred_str = "-".join([slot, pred_str])
                pred_set.add(pred_str)
        extracted_grd.append(grd_set)
        extracted_pred.append(pred_set)
        data_id = dataset.id_data_map[which][idx]
        data_ids.append(data_id)

    joint_acc, slot_acc = cal_metric(extracted_grd, extracted_pred, all_data_idx, id_data_map, bad_set)
    print("Joint ACC: %.4f, Slot ACC: %.4f" %(joint_acc, slot_acc))

    if joint_acc >= best_joint_acc and conf["dump_res"]:
        if conf["use_lbp"]:
            output_filepath = "./all_prediction_redst_gru_%s.json" %(which)
        else:
            output_filepath = "./all_prediction_redst_gru_%s_noprop.json" %(which)
        print("Get the best joint_acc, dump the result to file: %s" %(output_filepath))
        result = {}
        for grd, pred, data_id in zip(extracted_grd, extracted_pred, data_ids):
            ID, turn_id = data_id.split("-")
            if ID not in result:
                result[ID] = {turn_id: {"turn_belief": list(grd), "pred_bs_ptr": list(pred)}}
            else:
                result[ID][turn_id] = {"turn_belief": list(grd), "pred_bs_ptr": list(pred)}
        json.dump(result, open(output_filepath, "w"), indent=4)
            
    return joint_acc, slot_acc

# after each epoch, we perform inference on the training set to update the last_belief_pred
def pred_train_next(model, dataset, device, conf, which="train"):
    if which == "train":
        the_loader = dataset.train_loader
    if which == "dev":
        the_loader = dataset.dev_loader
    if which == "test":
        the_loader = dataset.test_loader
    model.eval()
    #pbar = tqdm(enumerate(the_loader), total=len(the_loader))
    all_grd, all_pred, all_data_idx, cont_ids = [], [], [], []
    loss_print = 0
    #for batch_i, batch in pbar:
    for batch_i, batch in enumerate(the_loader):
        [last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, cont_id] = [each.to(device) for each in batch]
        losses, preds = model(last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief)

        cont_ids += cont_id.squeeze(-1).detach().cpu().numpy().tolist()
        each_pred = []
        for x in preds:
            x = x.detach().cpu() # [batch_size, label_num]
            x_idx = torch.argmax(x, dim=-1).numpy().tolist() # [batch_size] 
            each_pred.append(x_idx) 
        # each_pred: [num_slots, batch_size]
        each_pred = np.array(each_pred).transpose(1, 0).tolist()
        all_pred += each_pred

    # save the prediction of every training sample within the dict: id_pred_map
    slot_id_label_map, id_slot_map = dataset.slot_id_label_map, dataset.id_slot_map
    id_pred_map = {}
    for pred, idx in zip(all_pred, cont_ids):
        pred_set = set()
        for i, slot in sorted(id_slot_map.items(), key=lambda i: int(i[0])):
            i = int(i)
            pred_str = slot_id_label_map[slot][str(pred[i])]
            if pred_str != "none":
                pred_str = "-".join([slot, pred_str])
                pred_set.add(pred_str)
        id_pred_map[idx] = list(pred_set)

    return id_pred_map


def compute_acc(gold, pred, num_slot=30):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = num_slot
    ACC = num_slot - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def cal_metric(grd, pred, all_data_idx, id_data_map, bad_set):
    ttl, hit = 0, 0
    unseen_bad = 0
    all_accs = 0
    for i, j, idx in zip(grd, pred, all_data_idx):
        tmp_acc = compute_acc(i, j)
        all_accs += tmp_acc
        ttl += 1
        if i == j:
            data_id = id_data_map[idx]
            hit += 1
            if data_id in bad_set:
                #print(unseen_bad, i, j)
                unseen_bad += 1
            #else:
            #    hit += 1

    if ttl == 0:
        if hit == 0:
            joint_acc = 1
        else:
            joint_acc = 0
    else:
        joint_acc = hit/ttl

    slot_acc = all_accs / float(ttl)

    print("ttl/hit/unseen_bad: %d/%d/%d, %f/%f" %(ttl, hit, unseen_bad, joint_acc, unseen_bad/ttl))
    return joint_acc, slot_acc
    

def cal_metric_for_each(grd, pred):
    ttl, hit = 0, 0
    unseen_bad = 0
    for i, j in zip(grd, pred):
        ttl += 1 
        if i == j:
            hit += 1
    if ttl == 0:
        if hit == 0:
            acc = 1
        else:
            acc = 0
    else:
        acc = hit/ttl
    return acc
    

def main():
    conf = yaml.load(open("./config.yaml"))
    for k, v in sorted(conf.items(), key=lambda i: i[0]):
        print(k, v)
    print("------------ split ----------")
    train(conf)


if __name__ == "__main__":
    main()

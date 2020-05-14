import json
import yaml
import numpy as np
from seqeval.metrics import f1_score
from tqdm import tqdm
import torch
from torch.optim import lr_scheduler
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import BertAdam 
from pytorch_pretrained_bert.tokenization import BertTokenizer

from model import MyBERT
from helper import MultiwozTestData, MultiwozData
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def pred(testdata, model, device, input_file, output_file):
    id_class_map = {0:"generate", 1:["yes"], 2:["no"], 3:["don", "##tc", "##are"], 4:["none"]}
    predictions = []
    dataloader = testdata.test_loader

    cnt = 0
    for batch in dataloader:
        cnt += 1
        batch = tuple(t.to(device) for t in batch)
        input_ids, type_ids, labels, classes, mask, domain_slot, dial_name, turn_id = batch
        
        with torch.no_grad():
            token_logits, seq_logits = model(input_ids, type_ids, mask)
            
        input_ids = input_ids.detach().cpu().numpy()
        token_logits = token_logits.detach().cpu().numpy() #(32, 5)
        seq_logits = seq_logits.detach().cpu().numpy() #(32, 80, 6)
        
        seq_logits = np.argmax(seq_logits, axis=1) # (32,)
        token_logits = np.argmax(token_logits, axis=2) #(32, 80)
        
        masks = mask.cpu().numpy() #(32, 80)
    
        domain_slot = domain_slot.squeeze(-1).cpu().numpy()
        domain_slot = [testdata.id_to_slot[x] for x in domain_slot]
        
        dial_name = dial_name.squeeze(-1).cpu().numpy()
        dial_name = [testdata.id_to_dialog[x] for x in dial_name]
        turn_id = turn_id.squeeze(-1).cpu().numpy().tolist()
        
        if cnt % 100 == 0:
            print(cnt, ' batches processed')
        for sample_idx in range(len(turn_id)):
            # for each testing sample
            _class = seq_logits[sample_idx]
            #if _class == 4:
            #    continue
            
            if _class > 0:
                value = id_class_map[_class]
            else:
                _tokenl = token_logits[sample_idx]
                _ids = tokenizer.convert_ids_to_tokens(input_ids[sample_idx])
                _mask = mask[sample_idx]
                value = []
                #{"[PAD]":0, "[CLS]":1, "[SEP]":2, "S":3, "I":4, "O":5}
                i = 0
                found = False
                while i < 80:
                    if _mask[i] < 1:
                        break
                    if _tokenl[i] == 3:
                        value.append( _ids[i] )
                        i = i + 1
                        while i < 80:
                            if _tokenl[i] == 4:
                                value.append( _ids[i] )
                            else:
                                found = True
                                break
                            i= i + 1
                    if found:
                        break
                    i = i+1
                        
                if len(value) < 1:
                    continue
                    
            prediction = {}
            prediction['dial_name'] = dial_name[sample_idx]
            prediction['turn_id'] = turn_id[sample_idx]
            prediction['pred_entry'] = { domain_slot[sample_idx] : value }
            predictions.append(prediction)
            
    # merge predicted entries into turns, dialogs
    pred_new_dials = {}
    for i in range(len(predictions)):
        prediction = predictions[i]
    
        if prediction['dial_name'] not in pred_new_dials:
            pred_new_dials[prediction['dial_name']] = {}
    
        if prediction['turn_id'] not in pred_new_dials[prediction['dial_name']]:
            pred_new_dials[prediction['dial_name']][prediction['turn_id']] = []
    
        valuestr = ''
        for k, v in prediction['pred_entry'].items():
            valuestr = k + '-' + ' '.join(v)

        pred_new_dials[prediction['dial_name']][prediction['turn_id']].append(valuestr)
    
    print("predictions len:", len(predictions))
    print("pred_new_dials len:", len(pred_new_dials))
    print(list(pred_new_dials.keys())[0], pred_new_dials[list(pred_new_dials.keys())[0]])

    # update ground truth file format
    new_dials = {}
    with open(input_file) as f:
        dials = json.load(f)
        for dial_name in dials.keys():
            #print(dial_name)
            dial = dials[dial_name]
            new_dial = {}
            for ti in dial.keys():
                turn = dial[ti]
                #print(ti, turn)
                turn_label, belief_label = [], []
                label_dict = turn['label_dict']
                belief_dict = turn['belif_dict']
                
                for key, value in label_dict.items():
                    value = ' '.join(value)
                    if value != 'none':
                        turn_label.append( key + '-' + value)
                        
                for key, value in belief_dict.items():
                    value = ' '.join(value)
                    if value != 'none':
                        belief_label.append( key + '-' + value)
                new_dial[ti] =  {'label_list':turn_label, 'belief_list':belief_label}
            new_dials[dial_name] = new_dial
    print("new_dials len:", len(new_dials))
    print(list(new_dials.keys())[0], new_dials[list(new_dials.keys())[0]])

    # merge two predicted format and ground truth into one json
    def to_dict(l):
        dic = {}
        for i in l:
            dic[i.split('-')[0]] = i.split('-')[1]
        return dic
    
    def merge(pred_list, belief_list):
        new_turn_belief_dic = {}
        for k, v in to_dict(belief_list).items():
            new_turn_belief_dic[k] = v
        for k, v in to_dict(pred_list).items():
            new_turn_belief_dic[k] = v
        return [ str(k) +'-'+ str(v) for k, v in new_turn_belief_dic.items()]
    
    dials_with_pred_truth = {}
    for dial_name in new_dials.keys():
        dial = new_dials[dial_name]
        new_dial = {}
        last_belief=[]
        for ti in dial.keys():
            new_turn = {}
            new_turn['true_labels'] = dial[ti]['label_list']
            new_turn['true_beliefs'] = dial[ti]['belief_list']
            new_turn['last_beliefs'] = last_belief
            #print(type(ti), type(list(pred_new_dials[dial_name].keys())[0]))
            if ti in pred_new_dials[dial_name]:
                new_turn['pred_labels'] = pred_new_dials[dial_name][int(ti)]
            else:
                new_turn['pred_labels'] = []
    
            predicted_turn_belief = merge(new_turn['pred_labels'] , last_belief)
            new_turn['predicted_beliefs'] = predicted_turn_belief
            last_belief = predicted_turn_belief
            new_dial[ti] =  new_turn
        dials_with_pred_truth[dial_name] = new_dial
    print(list(dials_with_pred_truth.keys())[0], dials_with_pred_truth[list(dials_with_pred_truth.keys())[0]])
    
    output = open(output_file, "w")
    json.dump(dials_with_pred_truth, output, indent=4)
    output.close()
    print('finished!')


def main():
    #model_path = "./params/model.th"
    #model_path = "./params/model_lizi.th"
    model_path = "./params/model_lizi_0.48928.th"
    model = MyBERT.from_pretrained("bert-base-uncased", num_token_labels=6, num_seq_labels=5)
    model.load_state_dict(torch.load(model_path).state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print("\n\nstart to predict dev set:")
    testdata = MultiwozTestData(which="dev")
    #pred(testdata, model, device, "data/processed_dev_dials.json", "results/turn_belief_pred_dev.json")
    #pred(testdata, model, device, "data/processed_dev_dials.json", "results/turn_belief_pred_lizi_dev.json")
    pred(testdata, model, device, "data/processed_dev_dials.json", "results/turn_belief_pred_lizi_0.48928_dev.json")

    print("\n\nstart to predict test set:")
    testdata = MultiwozTestData(which="test")
    #pred(testdata, model, device, "data/processed_test_dials.json", "results/turn_belief_pred_test.json")
    #pred(testdata, model, device, "data/processed_test_dials.json", "results/turn_belief_pred_lizi_test.json")
    pred(testdata, model, device, "data/processed_test_dials.json", "results/turn_belief_pred_lizi_0.48928_test.json")

    print("\n\nstart to predict train set:")
    testdata = MultiwozTestData(which="train")
    #pred(testdata, model, device, "data/processed_train_dials.json", "results/turn_belief_pred_train.json")
    #pred(testdata, model, device, "data/processed_train_dials.json", "results/turn_belief_pred_lizi_train.json")
    pred(testdata, model, device, "data/processed_train_dials.json", "results/turn_belief_pred_lizi_0.48928_train.json")


if __name__ == "__main__":
    main()

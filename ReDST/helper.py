import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

class DSTDataset(Dataset):
    def __init__(self, ori_data, conf, slot_label_id_map, slot_id_map, label_id_map, data_id_map=None, id_pred_map=None):
       # note that the length of label_id_map is not identical with all the number of classifier labels;
        # classifier labels include num_slot of "none";
        self.data = ori_data
        self.conf = conf
        self.slot_label_id_map = slot_label_id_map
        self.label_id_map = label_id_map
        self.slot_id_map = slot_id_map
        self.num_slot = len(slot_id_map)
        self.num_label = len(label_id_map)
        self.id_pred_map = id_pred_map # mapping from the train id to the corresponding prediction
        self.data_id_map = data_id_map # mapping from the "ID-turn_id" to an idx, to filter out the out-of-scheme testing samples


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        each = self.data[idx]
        last_belief_grd = self.convert_bs2id(each, "last_belief") 
        turn_belief_grd = self.convert_bs2id(each, "turn_belief")

        cont_id = torch.LongTensor([each["id_"]])

        if self.conf["input_type"] == "binary":
            last_belief_pred = self.convert_bs2id(each, "last_belief") 
            turn_belief_pred = self.convert_bs2id(each, "pred_bs_ptr")
        if self.conf["input_type"] == "prob":
            last_belief_pred = self.convert_bs2prob(each, "pred_last_bs_score_map") 
            turn_belief_pred = self.convert_bs2prob(each, "pred_bs_score_map")

        # use the model's prediction on last step to initialize the last_belief_pred, no the bert's output
        if self.id_pred_map is not None:
            last_id = each["last_id"]
            if last_id != -1:
                last_belief_pred = self.convert_pred_bs2id(self.id_pred_map[last_id])

        current_belief = self.get_class_label(each)

        if self.data_id_map:
            data_id = "-".join([str(each["ID"]), str(each["turn_id"])])
            data_idx = self.data_id_map[data_id]
            data_idx = torch.LongTensor([data_idx])
            return last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, data_idx 
        else:
            return last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, cont_id


    def get_class_label(self, each):
        # note that 0 is the id of "none" for each slot
        class_label = torch.zeros(self.num_slot).long()
        for x in each["current_belief"]:
            x = x.split("-")
            slot = "-".join(x[0:2])
            if slot not in self.slot_id_map:
                continue
            slot_id = int(self.slot_id_map[slot])

            value = x[-1]
            if value not in self.slot_label_id_map[slot]:
                continue
            value_id = int(self.slot_label_id_map[slot][value])

            class_label[slot_id] = value_id

        return class_label


    def convert_pred_bs2id(self, slot_value_list):
        res_ids = torch.zeros(self.num_label).float()
        for x in slot_value_list:
            x_id = int(self.label_id_map[x])
            res_ids[x_id] = 1
        return res_ids


    def convert_bs2id(self, each, key):
        res_ids = torch.zeros(self.num_label).float()
        for x in each[key]:
           if x in self.label_id_map:
               x_id = int(self.label_id_map[x])
               res_ids[x_id] = 1
        return res_ids


    def convert_bs2prob(self, each, key):
        res_ids = torch.zeros(self.num_label).float()
        for slot, value_score in each[key].items():
           value = value_score["value"]
           # we use the average of all words' prob
           score = sum(value_score["score"]) / len(value_score["score"])
           x = "-".join([slot, value])
           if x in self.label_id_map:
               x_id = self.label_id_map[x]
               res_ids[x_id] = score
        return res_ids


class MultiwozDSTData(Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.train_data, self.str_id_map_train, self.id_str_map_train = self.load_data("train")
        self.dev_data, self.str_id_map_dev, self.id_str_map_dev = self.load_data("dev")
        self.test_data, self.str_id_map_test, self.id_str_map_test = self.load_data("test")

        self.slot_label_id_map, self.slot_id_label_map, self.slot_id_map, self.id_slot_map, self.label_id_map, self.id_label_map, self.num_label_list = self.get_slot_label_id_map() 
        self.bad_set, self.data_id_map, self.id_data_map = self.check_dev_test_labels()

        self.adj, self.ent2attr_adj, self.attr2ent_adj = self.load_graph()
        self.init_dataset()

     
    def init_dataset(self, train_id_pred_map=None, dev_id_pred_map=None, test_id_pred_map=None):
        self.train_set = DSTDataset(self.train_data, self.conf, self.slot_label_id_map, self.slot_id_map, self.label_id_map, id_pred_map=train_id_pred_map)
        self.train_loader = DataLoader(self.train_set, batch_size=self.conf["batch_size"], shuffle=True, num_workers=10)
        self.dev_set = DSTDataset(self.dev_data, self.conf, self.slot_label_id_map, self.slot_id_map, self.label_id_map, data_id_map=self.data_id_map["dev"], id_pred_map=dev_id_pred_map)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.conf["batch_size"], shuffle=False, num_workers=0)
        self.test_set = DSTDataset(self.test_data, self.conf, self.slot_label_id_map, self.slot_id_map, self.label_id_map, data_id_map=self.data_id_map["test"], id_pred_map=test_id_pred_map)
        self.test_loader = DataLoader(self.test_set, batch_size=self.conf["batch_size"], shuffle=False, num_workers=0)


    def load_graph(self):
        all_graph_nodes = json.load(open("./data/lbp_graph_allnode_id_map.json"))
        edges = json.load(open("./data/lbp_graph_edge.json"))
        node_num = len(self.label_id_map)

        value_slotvalue_map = {}
        for label, id in self.label_id_map.items():
            domain, slot, value = label.split("-")
            if value not in value_slotvalue_map:
                value_slotvalue_map[value] = {label}
            else:
                value_slotvalue_map[value].add(label)

        adj = np.zeros([node_num, node_num])
        ent2attr_adj = np.zeros([node_num, node_num])
        attr2ent_adj = np.zeros([node_num, node_num])
        for venue, attrs in edges.items():
            if venue not in value_slotvalue_map: # test/dev has unseen venues
                continue
            for each_label1 in value_slotvalue_map[venue]:
                label1_id = int(self.label_id_map[each_label1])
                for each_attr in attrs:
                    if each_attr not in value_slotvalue_map: # test/dev has unseen attrs
                        continue
                    for each_label2 in value_slotvalue_map[each_attr]:
                        label2_id = int(self.label_id_map[each_label2])
                        adj[label1_id][label2_id] = 1
                        adj[label2_id][label1_id] = 1
                        ent2attr_adj[label2_id][label1_id] = 1
                        attr2ent_adj[label1_id][label2_id] = 1
        np.save("./data/lbp_graph_adj", adj)
        np.save("./data/lbp_graph_adj_ent2attr", ent2attr_adj)
        np.save("./data/lbp_graph_adj_attr2ent", attr2ent_adj)
        return adj, ent2attr_adj, attr2ent_adj


    def load_data(self, which):
        #data = json.load(open("./all_prediction_TRADE_%s.json" %(which)))
        data = json.load(open("./all_prediction_TRADE_preprocessed_%s_lizi.json" %(which)))
        #data = json.load(open("../turn_belief_gen/results/turn_belief_pred_%s.json" %(which)))
        result = []
        str_id_map = {}
        id_str_map = {}
        for ID, res in data.items():
            last_id = -1
            for turn_id, cont in sorted(res.items(), key=lambda i: int(i[0])):
                cont["ID"] = ID
                cont["turn_id"] = turn_id
                str_ = "%s_%s" %(ID, turn_id)
                id_ = len(str_id_map)
                str_id_map[str_] = id_
                id_str_map[id_] = str_
                cont["id_"] = id_
                # let each turn of uttr remember its predecessor uttr, if one uttr is the first sentence of a dial, just set as -1 and use itself.
                cont["last_id"] = last_id
                result.append(cont)
                last_id = id_
        return result, str_id_map, id_str_map


    def check_dev_test_labels(self):
        bad_set = {}
        data_id_map, id_data_map = {}, {}
        for which in ["dev", "test"]:
            if which == "dev":
                data = self.dev_data
            if which == "test":
                data = self.test_data
            ttl = len(data)
            bad = 0
            bad_id_set = set()
            tmp_data_id_map, tmp_id_data_map = {}, {}
            bad_set_out = open("%s_bad_set.json" %(which), "w")
            for each in data:
                ID = str(each["ID"])
                turn_id = str(each["turn_id"])
                each_id = "-".join([ID, turn_id])
                idx = len(tmp_data_id_map)
                tmp_data_id_map[each_id] = idx 
                tmp_id_data_map[idx] = each_id
                for x in each["current_belief"]:
                    x = x.split("-")
                    slot = "-".join(x[0:2])
                    if slot not in self.slot_id_map:
                        continue
                    slot_id = int(self.slot_id_map[slot])

                    value = x[-1]
                    if value not in self.slot_label_id_map[slot]:
                        bad += 1
                        bad_set_out.write("%d %s %s\n" %(bad, slot, value))
                        #print(ID, turn_id, slot, value)
                        bad_id_set.add(each_id)
                        break
            bad_set_out.close()
            bad_set[which] = list(bad_id_set)
            data_id_map[which] = tmp_data_id_map
            id_data_map[which] = tmp_id_data_map
            print("%s: Including unseen value samples: %d/%d: %f" %(which, bad, ttl, bad/ttl))
        return bad_set, data_id_map, id_data_map


    def get_slot_label_id_map(self):
        slot_label_file_path_1 = "./data/slot_label_id_map.json"
        slot_label_file_path_2 = "./data/slot_id_label_map.json"
        label_file_path_1 = "./data/label_id_map.json"
        label_file_path_2 = "./data/id_label_map.json"
        slot_file_path_1 = "./data/slot_id_map.json"
        slot_file_path_2 = "./data/id_slot_map.json"
      
        print('generate label/slot id map file')
        slot_label_id_map, slot_id_label_map = {}, {}
        slot_id_map, id_slot_map = {}, {}
        label_id_map, id_label_map = {}, {}
        for each in self.train_data:
            for k in ["turn_belief", "current_belief", "last_belief"]:
                for each_label in each[k]:
                    domain = each_label.split("-")[0]
                    if domain not in EXPERIMENT_DOMAINS:
                        continue 

                    if each_label not in label_id_map:
                        id = str(len(label_id_map))
                        label_id_map[each_label] = id
                        id_label_map[id] = each_label

                    each_label = each_label.split("-")
                    slot = "-".join(each_label[0:2])
                    value = each_label[-1]
                    if slot not in slot_label_id_map:
                        # add the "none" label for each slot's values
                        slot_label_id_map[slot] = {"none": "0", value: "1"}
                        slot_id_label_map[slot] = {"0": "none", "1": value}
                    else:
                        if value not in slot_label_id_map[slot]:
                            id = str(len(slot_label_id_map[slot]))
                            slot_label_id_map[slot][value] = id 
                            slot_id_label_map[slot][id] = value

                    if slot not in slot_id_map:
                        id = str(len(slot_id_map))
                        slot_id_map[slot] = id
                        id_slot_map[id]  = slot

        json.dump(slot_label_id_map, open(slot_label_file_path_1, "w"), indent=4)
        json.dump(slot_id_label_map, open(slot_label_file_path_2, "w"), indent=4)
        json.dump(label_id_map, open(label_file_path_1, "w"), indent=4)
        json.dump(id_label_map, open(label_file_path_2, "w"), indent=4)
        json.dump(slot_id_map, open(slot_file_path_1, "w"), indent=4)
        json.dump(id_slot_map, open(slot_file_path_2, "w"), indent=4)

        # a list of number of labels for each slot
        num_label_list = []
        for id, slot in sorted(id_slot_map.items(), key=lambda i: int(i[0])):
            num_label_list.append(len(slot_label_id_map[slot]))

        return slot_label_id_map, slot_id_label_map, slot_id_map, id_slot_map, label_id_map, id_label_map, num_label_list

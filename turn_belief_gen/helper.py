import json
import random
import torch
import numpy as np
from seqeval.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
random.seed(2019)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_accuracy1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


class myDataset(Dataset):
    def __init__(self, ori_data,max_len, vocab):
        self.data = ori_data
        self.max_len = max_len
        self.vocab = vocab
        self.label_id_map = {"[PAD]":0, "[CLS]":1, "[SEP]":2, "S":3, "I":4, "O":5}
        self.class_id_map = {"generate":0, "yes":1, "no":2, "doncare":3, "none":4}


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        sample = self.data[idx]
        max_len = self.max_len
        
        if len(sample['uttr_tokens']) <= max_len:
            input_ids = [self.vocab[x] for x in sample['uttr_tokens']] 
            actual_len = len(input_ids)

            input_ids += [0] * (max_len - actual_len)

            labels = [self.label_id_map[x] for x in sample['lables']]
            labels += [0] * (max_len - actual_len)

            _class = [self.class_id_map[ sample['class'] ]]

            assert len(input_ids) == len(labels)

            attention_mask = [1] * actual_len + [0] * (max_len-actual_len)

            type_ids = [0] * 4 + [1] * (actual_len-4) + [0]*(max_len-actual_len)
        else:
            input_ids = [self.vocab[x] for x in sample['uttr_tokens'][0:max_len-1]] + [self.vocab["[SEP]"]]
            labels = [self.label_id_map[x] for x in sample['lables'][0:max_len-1]] + [self.label_id_map["[SEP]"]]
            type_ids = [0] * 4 + [1] * (max_len-4)
            _class = [self.class_id_map[ sample['class'] ]]
            attention_mask = [1] * max_len
        
        input_ids = torch.Tensor(input_ids).long()
        type_ids = torch.Tensor(type_ids).long()
        labels = torch.Tensor(labels).long()
        _class = torch.Tensor(_class).long()
        attention_mask = torch.Tensor(attention_mask).long()

        return input_ids, type_ids, labels, _class, attention_mask


class MultiwozData(Dataset):
    def __init__(self):
        self.train_data, maxlen_train = self.load_data("./data/new_train_dials.json")
        self.test_data, maxlen_test = self.load_data("./data/new_test_dials.json")

        max_len = 80#max(maxlen_train, maxlen_test)
        print("Train set len: %d" %(len(self.train_data)))
        print("Test set len: %d" %(len(self.test_data)))
        print("Max seq len: %d" % (max_len))
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        vocab = tokenizer.vocab
        
        self.train_set = myDataset(self.train_data, max_len, vocab)
        self.train_loader = DataLoader(self.train_set, batch_size=32, shuffle=True, num_workers=10)
        
        self.test_set = myDataset(self.test_data, max_len, vocab)
        self.test_loader = DataLoader(self.test_set, batch_size=32, shuffle=False, num_workers=0)


    def load_data(self, path):
        data = json.load(open(path))
        res = []
        LENS = []
        for dial_name in data.keys():
            dial = data[dial_name]
            for ti in dial.keys():
                for slot in dial[ti].keys():
                    sample = {}
                    sample["dial_name"] = dial_name
                    sample["turn_id"] = ti
                    sample["domain_slot"] = slot
                    sample["uttr_tokens"] = dial[ti][slot]["uttr_tokens"]
                    
                    sample["lables"] = dial[ti][slot]["lables"]
                    sample["class"] = dial[ti][slot]["class"]
                    #if sample["class"] == 'none' and random.random() < 0.8:
                    #    continue
                    LENS.append(len(sample["uttr_tokens"]))
                    res.append(sample)
        return res, max(set(LENS))


class myDatasetTest(Dataset):
    def __init__(self, ori_data,max_len, vocab):
        self.data = ori_data
        self.max_len = max_len
        self.vocab = vocab
        self.label_id_map = {"[PAD]":0, "[CLS]":1, "[SEP]":2, "S":3, "I":4, "O":5}
        self.class_id_map = {"generate":0, "yes":1, "no":2, "doncare":3, "none":4}


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        sample = self.data[idx]
        max_len = self.max_len
        
        if len(sample['uttr_tokens']) <= max_len:
            input_ids = [self.vocab[x] for x in sample['uttr_tokens']] 
            actual_len = len(input_ids)

            input_ids += [0] * (max_len - actual_len)
            labels = [self.label_id_map[x] for x in sample['lables']]
            labels += [0] * (max_len - actual_len)

            _class = [self.class_id_map[ sample['class'] ]]
            assert len(input_ids) == len(labels)

            attention_mask = [1] * actual_len + [0] * (max_len-actual_len)
            type_ids = [0] * 4 + [1] * (actual_len-4) + [0]*(max_len-actual_len)
        else:
            input_ids = [self.vocab[x] for x in sample['uttr_tokens'][0:max_len-1]] + [self.vocab["[SEP]"]]
            labels = [self.label_id_map[x] for x in sample['lables'][0:max_len-1]] + [self.label_id_map["[SEP]"]]
            type_ids = [0] * 4 + [1] * (max_len-4)
            _class = [self.class_id_map[ sample['class'] ]]
            attention_mask = [1] * max_len
        
        input_ids = torch.Tensor(input_ids).long()
        type_ids = torch.Tensor(type_ids).long()
        labels = torch.Tensor(labels).long()
        _class = torch.Tensor(_class).long()
        attention_mask = torch.Tensor(attention_mask).long()
        
        domain_slot = torch.Tensor([sample["domain_slot"]]).long()
        dial_name = torch.Tensor([sample["dial_name"]]).long()
        turn_id = torch.Tensor([sample["turn_id"]]).long()

        return input_ids, type_ids, labels, _class, attention_mask, domain_slot, dial_name, turn_id


class MultiwozTestData(Dataset):
    def __init__(self, which="test"):
        self.SLOTS_to_id = {'hotel price': 0, 'hotel type': 1, 'hotel parking': 2, 'hotel stay': 3, 'hotel day': 4, 'hotel people': 5, 'hotel area': 6, 'hotel stars': 7, 'hotel internet': 8, 'train destination': 9, 'train day': 10, 'train departure': 11, 'train arrive': 12, 'train people': 13, 'train leave': 14, 'attraction area': 15, 'restaurant food': 16, 'restaurant price': 17, 'restaurant area': 18, 'attraction name': 19, 'restaurant name': 20, 'attraction type': 21, 'hotel name': 22, 'taxi leave': 23, 'taxi destination': 24, 'taxi departure': 25, 'restaurant time': 26, 'restaurant day': 27, 'restaurant people': 28, 'taxi arrive': 29}
        
        self.id_to_slot = {0: 'hotel price', 1: 'hotel type', 2: 'hotel parking', 3: 'hotel stay', 4: 'hotel day', 5: 'hotel people', 6: 'hotel area', 7: 'hotel stars', 8: 'hotel internet', 9: 'train destination', 10: 'train day', 11: 'train departure', 12: 'train arrive', 13: 'train people', 14: 'train leave', 15: 'attraction area', 16: 'restaurant food', 17: 'restaurant price', 18: 'restaurant area', 19: 'attraction name', 20: 'restaurant name', 21: 'attraction type', 22: 'hotel name', 23: 'taxi leave', 24: 'taxi destination', 25: 'taxi departure', 26: 'restaurant time', 27: 'restaurant day', 28: 'restaurant people', 29: 'taxi arrive'}

        self.dialg_to_id = {}
        self.id_to_dialog = {}
        if which == "train":
            datafile = "./data/new_train_dials.json"
        if which == "dev":
            datafile = "./data/new_dev_dials.json"
        if which == "test":
            datafile = "./data/new_test_dials.json"

        self.test_data, maxlen_test = self.load_data(datafile)

        max_len = 80 #max(maxlen_train, maxlen_test)
        print("Test set len: %d" %(len(self.test_data)))
        print("Max seq len: %d" % (max_len))
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.vocab = tokenizer.vocab
        
        self.test_set = myDatasetTest(self.test_data, max_len, self.vocab)
        self.test_loader = DataLoader(self.test_set, batch_size=500, shuffle=False, num_workers=10)


    def load_data(self, path):
        data = json.load(open(path))
        dialg_id = 0
        res = []
        LENS = []
        for dial_name in data.keys():
            dial = data[dial_name]
            for ti in dial.keys():
                for slot in dial[ti].keys():
                    sample = {}
                    
                    if dial_name not in self.dialg_to_id.keys():
                        self.dialg_to_id[dial_name] = dialg_id
                        dialg_id += 1
                    sample["dial_name"] = self.dialg_to_id[dial_name]
                    
                    sample["turn_id"] = int(ti)
                    sample["domain_slot"] = self.SLOTS_to_id[slot]
                    sample["uttr_tokens"] = dial[ti][slot]["uttr_tokens"]
                    
                    sample["lables"] = dial[ti][slot]["lables"]
                    sample["class"] = dial[ti][slot]["class"]
                    #if sample["class"] == 'none' and random.random() < 0.8:
                    #    continue
                    LENS.append(len(sample["uttr_tokens"]))
                    res.append(sample)
                    
        for key in self.dialg_to_id.keys():
            self.id_to_dialog[self.dialg_to_id[key]] = key
            
        return res, max(set(LENS))

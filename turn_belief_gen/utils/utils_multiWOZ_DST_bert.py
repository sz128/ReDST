import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm
import os
import pickle
from random import shuffle

from .fix_label import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        #self.last_domain = data_info['last_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']

        self.current_belief = data_info['current_belief']
        self.current_gating_label = data_info['current_gating_label']
        self.current_generate_y = data_info["current_generate_y"]

        self.turn_belief = data_info['turn_belief']
        self.turn_gating_label = data_info['turn_gating_label']
        self.turn_generate_y = data_info["turn_generate_y"]

        self.last_belief = data_info['last_belief']
        self.last_gating_label = data_info['last_gating_label']
        self.last_generate_y = data_info["last_generate_y"]

        self.turn_uttr = data_info['turn_uttr']
        self.turn_uttr_bert = data_info['turn_uttr_bert']
        self.type_ids = data_info['type_ids']

        self.system_acts_sent = data_info["system_acts_sent"]
        self.system_acts_sent_bert = data_info["system_acts_sent_bert"]
        self.system_acts_sent_type_ids = data_info['system_acts_sent_type_ids']

        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        #last_domain = self.preprocess_domain(self.last_domain[index])

        turn_gating_label = self.turn_gating_label[index]
        turn_belief = self.turn_belief[index]
        turn_generate_y = self.turn_generate_y[index]
        turn_generate_y = self.preprocess_slot(turn_generate_y, self.trg_word2id)

        current_gating_label = self.current_gating_label[index]
        current_belief = self.current_belief[index]
        current_generate_y = self.current_generate_y[index]
        current_generate_y = self.preprocess_slot(current_generate_y, self.trg_word2id)

        last_gating_label = self.last_gating_label[index]
        last_belief = self.last_belief[index]
        last_generate_y = self.last_generate_y[index]
        last_generate_y = self.preprocess_slot(last_generate_y, self.trg_word2id)

        context = self.dialog_history[index] 
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]
        turn_uttr = self.preprocess(self.turn_uttr[index], self.src_word2id)
        turn_uttr_bert = self.preprocess_bert(self.turn_uttr_bert[index])
        tmp = self.type_ids[index]
        turn_uttr_bert_type_ids = torch.Tensor([0] * tmp[0] + [1] * tmp[1])

        system_acts_sent = self.preprocess(self.system_acts_sent[index], self.src_word2id)
        system_acts_sent_bert = self.preprocess_bert(self.system_acts_sent_bert[index])
        system_acts_sent_bert_type_ids = torch.Tensor([0] * self.system_acts_sent_type_ids[index][0])

        turn_uttr_plain = self.turn_uttr[index]
        
        item_info = {
            "ID":ID, 
            "turn_id":turn_id,
            "turn_domain":turn_domain,
            #"last_domain":last_domain,

            "turn_belief":turn_belief,
            "turn_gating_label":turn_gating_label,
            "turn_generate_y":turn_generate_y,

            "current_belief":current_belief,
            "current_gating_label":current_gating_label,
            "current_generate_y":current_generate_y,

            "last_belief":last_belief,
            "last_gating_label":last_gating_label,
            "last_generate_y":last_generate_y,

            "context":context,
            "context_plain":context_plain,
            "turn_uttr":turn_uttr,
            "turn_uttr_bert":turn_uttr_bert,
            "turn_uttr_bert_type_ids":turn_uttr_bert_type_ids,
            "turn_uttr_plain":turn_uttr_plain,

            "system_acts_sent": system_acts_sent,
            "system_acts_sent_bert": system_acts_sent_bert,
            "system_acts_sent_bert_type_ids": system_acts_sent_bert_type_ids,
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_bert(self, sequence):
        """Converts words to ids."""
        story = tokenizer.convert_tokens_to_ids([x for x in sequence.split()])
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        if len(turn_domain) == 0:
            return None
        domains = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        return domains[turn_domain]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    # note that bert padding 0
    def merge_bert(sequences, type_ids):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.zeros(len(sequences), max_len).long()
        padded_type_ids = torch.zeros(len(type_ids), max_len).long()
        for i, (seq, type_id) in enumerate(zip(sequences, type_ids)):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            padded_type_ids[i, :end] = type_id[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        padded_type_ids = padded_type_ids.detach()
        return padded_seqs, padded_type_ids

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)

        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    #data.sort(key=lambda x: len(x['turn_uttr']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    turn_uttr_src_seqs, turn_uttr_src_lengths = merge(item_info['turn_uttr'])
    turn_uttr_src_bert_seqs, turn_uttr_src_bert_type_ids = merge_bert(item_info['turn_uttr_bert'], item_info['turn_uttr_bert_type_ids'])
    turn_uttr_mask = (turn_uttr_src_seqs != 1) # generate mask for input sequence

    system_acts_sent_seq, system_acts_sent_seq_lengths = merge(item_info['system_acts_sent'])
    system_acts_sent_bert_seqs, system_acts_sent_bert_type_ids = merge_bert(item_info['system_acts_sent_bert'], item_info['system_acts_sent_bert_type_ids'])
    system_acts_sent_mask = (system_acts_sent_seq != 1) # generate mask for input sequence

    turn_y_seqs, turn_y_lengths = merge_multi_response(item_info["turn_generate_y"])
    turn_gating_label = torch.tensor(item_info["turn_gating_label"])

    current_y_seqs, current_y_lengths = merge_multi_response(item_info["current_generate_y"])
    current_gating_label = torch.tensor(item_info["current_gating_label"])

    last_y_seqs, last_y_lengths = merge_multi_response(item_info["last_generate_y"])
    last_gating_label = torch.tensor(item_info["last_gating_label"])

    turn_domain = torch.tensor(item_info["turn_domain"])
    #last_domain = torch.tensor(item_info["last_domain"])

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        turn_uttr_src_seqs = turn_uttr_src_seqs.cuda()
        turn_uttr_mask = turn_uttr_mask.long().cuda()
        turn_uttr_src_bert_seqs = turn_uttr_src_bert_seqs.long().cuda()
        turn_uttr_src_bert_type_ids = turn_uttr_src_bert_type_ids.long().cuda()

        system_acts_sent_seq = system_acts_sent_seq.cuda()
        system_acts_sent_mask = system_acts_sent_mask.long().cuda()
        system_acts_sent_bert_seqs = system_acts_sent_bert_seqs.long().cuda()
        system_acts_sent_bert_type_ids = system_acts_sent_bert_type_ids.long().cuda() 

        turn_gating_label = turn_gating_label.cuda()
        turn_y_seqs = turn_y_seqs.cuda()
        turn_y_lengths = turn_y_lengths.cuda()

        current_gating_label = current_gating_label.cuda()
        current_y_seqs = current_y_seqs.cuda()
        current_y_lengths = current_y_lengths.cuda()

        last_gating_label = last_gating_label.cuda()
        last_y_seqs = last_y_seqs.cuda()
        last_y_lengths = last_y_lengths.cuda()

        turn_domain = turn_domain.cuda()
        #last_domain = last_domain.cuda()


    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["turn_uttr"] = turn_uttr_src_seqs
    item_info["turn_uttr_mask"] = turn_uttr_mask
    item_info["turn_uttr_len"] = turn_uttr_src_lengths
    item_info["turn_uttr_bert"] = turn_uttr_src_bert_seqs
    item_info["turn_uttr_bert_type_ids"] = turn_uttr_src_bert_type_ids

    item_info["system_acts_sent_seq"] = system_acts_sent_seq
    item_info["system_acts_sent_mask"] = system_acts_sent_mask
    item_info["system_acts_sent_len"] = system_acts_sent_seq_lengths 
    item_info["system_acts_sent_bert_seqs"] = system_acts_sent_bert_seqs
    item_info["system_acts_sent_bert_type_ids"] = system_acts_sent_bert_type_ids

    item_info["turn_gating_label"] = turn_gating_label
    item_info["turn_generate_y"] = turn_y_seqs
    item_info["turn_y_lengths"] = turn_y_lengths

    item_info["current_gating_label"] = current_gating_label
    item_info["current_generate_y"] = current_y_seqs
    item_info["current_y_lengths"] = current_y_lengths

    item_info["last_gating_label"] = last_gating_label
    item_info["last_generate_y"] = last_y_seqs
    item_info["last_y_lengths"] = last_y_lengths

    item_info["turn_domain"] = turn_domain
    #item_info["last_domain"] = last_domain

    return item_info

def read_langs(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {} 
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first 
        for dial_dict in dials:
            if (args["all_vocab"] or dataset=="train") and training:
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')

        lang.index_words("[CLS] [SEP]", 'utter')

        # determine training data ratio, default is 100%
        if training and dataset=="train" and args["data_ratio"]!=100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]
        
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            last_belief_list = []
            last_gating_label = [gating_dict["none"] for slot in SLOTS]
            last_generate_y =["none" for slot in SLOTS]
#            last_domain = ''

            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                type_ids = [0, 0]
                #turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                type_ids[0] = len(turn["system_transcript"].strip().split()) + 2
                type_ids[1] = len(turn["transcript"].strip().split()) + 1
                turn_uttr = "[CLS] " + turn["system_transcript"].strip() + " [SEP] " + turn["transcript"].strip() + " [SEP]"
                turn_uttr_bert = "[CLS] " + turn["system_transcript_ours"].strip() + " [SEP] " + turn["transcript_ours"].strip() + " [SEP]"
                turn_uttr_strip = turn_uttr.strip()

                system_acts_sent_type_ids = [0, 0]
                system_acts_sent_type_ids[0] = len(turn["system_acts_sent"].strip().split()) + 2
                system_acts_sent = "[CLS] " + turn["system_acts_sent"].strip() + " [SEP]"
                system_acts_sent_ours = "[CLS] " + turn["system_acts_sent_ours"].strip() + " [SEP]"

                dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()
                turn_belief_dict_ = fix_general_label_error(turn["belief_state"], False, SLOTS)
                turn_label_dict_ = fix_general_label_error(turn["turn_label"], True, SLOTS) ##

                turn_belief_dict = {}
                for k, v in turn_belief_dict_.items():
                    if str(v) != 'none':
                        turn_belief_dict[k] = v

                turn_label_dict = {}
                for k, v in turn_label_dict_.items():
                    if str(v) != 'none':
                        turn_label_dict[k] = v

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
                turn_label_list = [str(k)+'-'+str(v) for k, v in turn_label_dict.items()] ##

                if (args["all_vocab"] or dataset=="train") and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                class_label, generate_y, turn_generate_y, slot_mask, gating_label, turn_gating_label  = [], [], [], [], [], []
                for slot in slot_temp:
                    # for current turn utterance
                    if slot in turn_label_dict.keys():
                        turn_generate_y.append(turn_label_dict[slot])

                        if turn_label_dict[slot] == "dontcare":
                            turn_gating_label.append(gating_dict["dontcare"])
                        else:
                            turn_gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_label_dict[slot]):
                            max_value_len = len(turn_label_dict[slot])
                    else:
                        turn_generate_y.append("none")
                        turn_gating_label.append(gating_dict["none"])
                #print(turn_label_list) ['hotel-pricerange-cheap', 'hotel-type-hotel']
                #print(turn_generate_y)  ['cheap', 'hotel', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
                #print(turn_gating_label)  [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                #exit(1)

                for slot in slot_temp:
                    # for utterance history
                    if slot in turn_belief_dict.keys():
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])
                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])

                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "domains":dial_dict["domains"], 
                    "turn_domain":turn_domain,
                    "turn_id":turn_id, 
                    "dialog_history":source_text,

                    "current_belief":turn_belief_list,
                    "current_gating_label":gating_label,
                    "current_generate_y":generate_y,

                    "turn_belief":turn_label_list,
                    "turn_gating_label":turn_gating_label,
                    "turn_uttr":turn_uttr_strip,
                    "turn_uttr_bert": turn_uttr_bert,
                    "type_ids": type_ids,

                    "system_acts_sent": system_acts_sent,
                    "system_acts_sent_bert": system_acts_sent_ours,
                    "system_acts_sent_type_ids": system_acts_sent_type_ids,

                    'turn_generate_y':turn_generate_y,
                    "last_belief":last_belief_list,
     #               "last_domain":last_domain,
                    "last_gating_label":last_gating_label,
                    'last_generate_y':last_generate_y
                    }
                data.append(data_detail)
                last_belief_list = turn_belief_list
                last_gating_label = gating_label
                last_generate_y = generate_y
    #            last_domain = turn_domain
                
                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())
                
            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp


def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity):  
    if(type and args['fisher_sample']>0):
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 

    dataset = Dataset(data_info, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)

    if args["imbalance_sampler"] and type:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  # shuffle=type,
                                                  collate_fn=collate_fn,
                                                  sampler=ImbalancedDatasetSampler(dataset))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=type,
                                                  collate_fn=collate_fn)
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def prepare_data_seq(training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    #file_train = 'data/train_dials.json'
    #file_dev = 'data/dev_dials.json'
    #file_test = 'data/test_dials.json'
    file_train = 'data/new_train_dials.json'
    file_dev = 'data/new_dev_dials.json'
    file_test = 'data/new_test_dials.json'
    # Create saving folder
    if args['path']:
        folder_name = args['path'].rsplit('/', 2)[0] + '/'
    else:
        folder_name = 'save/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multi-woz/MULTIWOZ2.1/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len, slot_train = read_langs(file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)
        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle: 
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle: 
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle: 
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle: 
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    test_4d = []
    if args['except_domain']!="":
        pair_test_4d, _, _ = read_langs(file_test, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        test_4d  = get_seq(pair_test_4d, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    print(SLOTS_LIST[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])
    LANG = [lang, mem_lang]
    return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

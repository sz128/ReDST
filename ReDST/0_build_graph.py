import json
from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction"]
PATH = "../turn_belief_gen/data/multi-woz/MULTIWOZ2.1/"

def get_all_nodes():
    names_stat, attrs_stat = {}, {}
    data = json.load(open("./all_prediction_TRADE_preprocessed_train_lizi.json"))
    #data = json.load(open("../turn_belief_gen/results/turn_belief_pred_train.json"))
    for dial_id, res in data.items():
        for turn_id, cont in res.items():
            for each in cont["turn_belief"]+cont["current_belief"]:
                domain, slot, value = each.split("-")
                domain_slot = "-".join([domain, slot])
                if slot == "name":
                    if value not in names_stat:
                        names_stat[value] = 1
                    else:
                        names_stat[value] += 1
                else:
                    if value in {"yes", "no", "don"}:
                        value = each
                    if value not in attrs_stat:
                        attrs_stat[value] = 1
                    else:
                        attrs_stat[value] += 1
                        
    new_attrs_stat = {}
    for k, v in attrs_stat.items():
        if k not in names_stat:
            new_attrs_stat[k] = v
                        
    return names_stat, new_attrs_stat


def str_cmp(query, candi_sets, candi_strs, thresh=0.5):
    query_set = set(query.split())
    q_len = len(query_set)
    overlap_ratio = 0
    result = {}
    find = None
    for candi_set, candi_str in zip(candi_sets, candi_strs):
        c_len = len(candi_set)
        overlap_len = c_len + q_len - len(candi_set.union(query_set))
        if c_len > q_len:
            overlap_ratio = overlap_len / float(c_len)
        else:
            overlap_ratio = overlap_len / float(q_len)
        result[candi_str] = overlap_ratio
    for candi_str, ratio in sorted(result.items(), key=lambda i: i[1], reverse=True):
        if ratio >= thresh:
            find = candi_str
            #print(ratio, query, find)
        break
        
    return find
        

def build_graph(all_names_stat, all_attrs_stat, name_lb=1, attr_lb=1):
    # name_lb=10 means only keep those venue appearring more than 10 times in dataset
    # attr_lb=10 means only keep thos attr values appearring more than 10 times in dataset
    # simi_lb=0.5 means if a value in db has more than 50% overlapped words with a value 
    #     in dataset label, we will link this db value to that dataset value
    
    all_names, all_attrs = set(), set()
    all_names_candi_set, all_attrs_candi_set = [], []
    all_names_candi_str, all_attrs_candi_str = [], []
    for name, cnt in all_names_stat.items():
        if cnt >= name_lb:
            all_names.add(name)
            all_names_candi_set.append(set(name.split()))
            all_names_candi_str.append(name)
    for attr, cnt in all_attrs_stat.items():
        if cnt >= attr_lb:
            all_attrs.add(attr)
            all_attrs_candi_set.append(set(attr.split()))
            all_attrs_candi_str.append(attr)
            
    print("num of all_names in dataset labels: ", len(all_names))
    print("num of all_attrs in dataset labels: ", len(all_attrs))
    
    name_id_map, attr_id_map, all_attr_id_map = {}, {}, {} # save all the kept nodes
    name_map, attr_map = {}, {} # to map the str from db to dataset labels
    name_id, attr_id, all_attr_id = 0, 0, 0
    name_attr_link = {}
    find_pairs = {"name": {}, "attr": {}}
    for domain in EXPERIMENT_DOMAINS:
        try:
            inputfile = "%s%s_db.json" %(PATH, domain)
            data = json.load(open(inputfile))
        except:
            print(inputfile)
            exit(1)
        
        for each in data:
            name = ""
            if "name" in each:
                each["name"] = " ".join(tokenizer.tokenize(each["name"]))
                if each["name"] in all_names:
                    name = each["name"]
                else:
                    find = str_cmp(each["name"], all_names_candi_set, all_names_candi_str)
                    if find is not None:
                        name = find
                        if each["name"] not in find_pairs["name"]:
                            find_pairs["name"][each["name"]] = name
            if name != "":
                each_attrs = set() # keep all valid attrs
                for k, v in each.items():
                    if k in {"id", "introduction", "location", "price", "name"}:
                        continue
                    else:
                        v = " ".join(tokenizer.tokenize(v))
                        if v in {"yes", "no", "dontcare"}:
                            v = "-".join([domain, k, v])
                        if v in all_attrs:
                            each_attrs.add(v)
                        else:
                            find = str_cmp(v, all_attrs_candi_set, all_attrs_candi_str)
                            if find:
                                each_attrs.add(find)
                                if v not in find_pairs["attr"]:
                                    find_pairs["attr"][v] = find
                            
                if len(each_attrs) != 0:
                    if name not in name_id_map:
                        name_id_map[name] = name_id
                        name_id += 1
                        name_attr_link[name] = each_attrs
                    else:
                        name_attr_link[name].union(each_attrs)
                        
                    for x in each_attrs:
                        if x not in attr_id_map:
                            attr_id_map[x] = attr_id
                            attr_id += 1
                            
            each_attrs = set() # keep all valid attrs
            for k, v in each.items():
                if k in {"id", "introduction", "location", "price", "name"}:
                    continue
                else:
                    v = " ".join(tokenizer.tokenize(v))
                    if v in {"yes", "no", "dontcare"}:
                        v = "-".join([domain, k, v])
                    if v in all_attrs:
                        each_attrs.add(v)
                    else:
                        find = str_cmp(v, all_attrs_candi_set, all_attrs_candi_str)
                        if find is not None:
                            each_attrs.add(find)

            if len(each_attrs) != 0:
                for x in each_attrs:
                    if x not in all_attr_id_map:
                        all_attr_id_map[x] = all_attr_id
                        all_attr_id += 1
                            
    print("num find pairs: name/attr: ", len(find_pairs["name"]), len(find_pairs["attr"]))  
                
    return name_id_map, attr_id_map, all_attr_id_map, name_attr_link, find_pairs


def build_graph_on_db():
    name_id_map, attr_id_map, all_attr_id_map = {}, {}, {} # save all the kept nodes
    name_attr_link = {}
    name_id, attr_id, all_attr_id = 0, 0, 0
    for domain in EXPERIMENT_DOMAINS:
        try:
            inputfile = "%s%s_db.json" %(PATH, domain)
            data = json.load(open(inputfile))
        except:
            print(inputfile)
            exit(1)
        
        for each in data:
            name = ""
            if "name" in each:
                name = each["name"]
            if name != "":
                each_attrs = set() # keep all valid attrs
                for k, v in each.items():
                    if k in {"id", "introduction", "location", "price", "name"}:
                        continue
                    else:
                        if v in {"yes", "no", "dontcare"}:
                            v = "-".join([domain, k, v])
                        each_attrs.add(v)
                            
                if len(each_attrs) != 0:
                    if name not in name_id_map:
                        name_id_map[name] = name_id
                        name_id += 1
                        name_attr_link[name] = each_attrs
                    else:
                        name_attr_link[name].union(each_attrs)
                        
                    for x in each_attrs:
                        if x not in attr_id_map:
                            attr_id_map[x] = attr_id
                            attr_id += 1
                            
            each_attrs = set() # keep all valid attrs
            for k, v in each.items():
                if k in {"id", "introduction", "location", "price", "name"}:
                    continue
                else:
                    if v in {"yes", "no", "dontcare"}:
                        v = "-".join([domain, k, v])
                    each_attrs.add(v)

            if len(each_attrs) != 0:
                for x in each_attrs:
                    if x not in all_attr_id_map:
                        all_attr_id_map[x] = all_attr_id
                        all_attr_id += 1  
                    
    return name_id_map, attr_id_map, all_attr_id_map, name_attr_link


def main():
    all_names, all_attrs = get_all_nodes()
    # the number of venue names and attr values in dataset labels
    # print(len(all_names), len(all_attrs))

    # note that we only consider the venue name and attr
    # of domain "attraction", "hotel", and "restaurant" 
    name_id_map, attr_id_map, all_attr_id_map, name_attr_link, find_pairs = build_graph(all_names, all_attrs)
    print("venues names in db that appear in dataset labels:", len(name_id_map))
    print("attr values in db that appear in dataset labels", len(attr_id_map))
    print("all attr values in db (including those do not have a venue name):", len(all_attr_id_map))

    # get the final result
    nameattr_id_map = dict(name_id_map)
    for k, v in attr_id_map.items():
        if k in nameattr_id_map:
            print("error")
        else:
            nameattr_id_map[k] = len(nameattr_id_map)
    print("num of valid entities: ", len(name_id_map))
    print("num of valid attrs: ", len(attr_id_map))
    print("num of all nodes: ", len(nameattr_id_map))

    # save the result
    json.dump(name_id_map, open("./data/lbp_graph_name_id_map.json", "w"), indent=4)
    json.dump(attr_id_map, open("./data/lbp_graph_attr_id_map.json", "w"), indent=4)
    json.dump(nameattr_id_map, open("./data/lbp_graph_allnode_id_map.json", "w"), indent=4)
    for k, v in name_attr_link.items():
        name_attr_link[k] = list(v)
    json.dump(name_attr_link, open("./data/lbp_graph_edge.json", "w"), indent=4)
    
    # make statistics only based on db, do not filter by dataset labels
    #name_id_map_db, attr_id_map_db, all_attr_id_map, name_attr_link_db = build_graph_on_db()
    #print("num venue names in db: ", len(name_id_map_db))
    #print("num attr of venues in db: ", len(attr_id_map_db))
    #print("num of all attrs in db: ", len(all_attr_id_map))


if __name__ == "__main__":
    main()

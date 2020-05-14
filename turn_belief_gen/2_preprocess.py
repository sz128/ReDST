# generate tokenized sequence, pad it, obtain labels correspondingly
import json
import pprint
from pytorch_pretrained_bert import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

SLOTS = ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day', 'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-book people', 'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'attraction-name', 'restaurant-name', 'attraction-type', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure', 'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arriveby']

to_new_SLOTS = {'hotel-pricerange':'hotel-price', 'hotel-type':'hotel-type', 'hotel-parking':'hotel-parking', 'hotel-book stay':'hotel-stay', 'hotel-book day':'hotel-day', 'hotel-book people':'hotel-people', 'hotel-area':'hotel-area', 'hotel-stars':'hotel-stars', 'hotel-internet':'hotel-internet', 'train-destination':'train-destination', 'train-day':'train-day', 'train-departure':'train-departure', 'train-arriveby':'train-arrive', 'train-book people':'train-people', 'train-leaveat':'train-leave', 'attraction-area':'attraction-area', 'restaurant-food':'restaurant-food', 'restaurant-pricerange':'restaurant-price', 'restaurant-area':'restaurant-area', 'attraction-name':'attraction-name', 'restaurant-name':'restaurant-name', 'attraction-type':'attraction-type', 'hotel-name':'hotel-name', 'taxi-leaveat':'taxi-leave', 'taxi-destination':'taxi-destination', 'taxi-departure':'taxi-departure', 'restaurant-book time':'restaurant-time', 'restaurant-book day':'restaurant-day', 'restaurant-book people':'restaurant-people', 'taxi-arriveby':'taxi-arrive'}

def inside(sub_l, l):
    if len(sub_l) < 1:
        print('empty sub list')
        exit()
    for i in range(len(l)):
        if l[i] == sub_l[0]:
            if len(sub_l) <= len(l) - i:
                flag = 1
                for ii in range(len(sub_l)):
                    if i+ii < len(l) and sub_l[ii] != l[i+ii]:
                        flag = 0
                        break
                if flag == 1:
                    return True, i, i+len(sub_l)
    return False, -1, -1


def process(input_file, output_file):
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
                turn_content = {}
               
                for domain_slot in SLOTS:
                    # for each domain slot
                    domain_slot = to_new_SLOTS[domain_slot].replace('-', ' ')
                    
                    turn_content[domain_slot] = {}
                    
                    uttr_token = turn['turn_uttr']
                    uttr = ['[CLS]'] + tokenizer.tokenize(domain_slot) + ['[SEP]'] + uttr_token +['[SEP]']
                    token_lable = ['[CLS]', 'O','O','[SEP]']
                    uttr_lable = ['O']*len(uttr_token) + ['[SEP]']
                    
                    
                    if domain_slot in turn['label_dict'].keys():
                        value = turn['label_dict'][domain_slot]
                        if value == ["yes"]:
                            _class = 'yes'
                        elif value == ["no"]:
                            _class = 'no'
                        elif value == ["don","##tc","##are"]:
                            _class = 'doncare'
                        else:
                            _class = 'generate'
                        
                            flag, start, end = inside(value, uttr_token)
                            if flag:
                                uttr_lable[start] = 'S'
                                for i in range(start+1, end):
                                    uttr_lable[i] = 'I'
                    else:
                        _class = 'none'
    
                    token_lable = token_lable + uttr_lable
    
                    turn_content[domain_slot]['uttr_tokens'] = uttr
                    turn_content[domain_slot]['lables'] = token_lable
                    turn_content[domain_slot]['class'] = _class
    
                new_dial[ti] = turn_content
            
            new_dials[dial_name] = new_dial
        
    output = open(output_file, "w")
    json.dump(new_dials, output, indent=4)
    output.close()
    print('finished!')


def main():
    process("data/processed_train_dials.json", "data/new_train_dials.json")
    process("data/processed_dev_dials.json", "data/new_dev_dials.json")
    process("data/processed_test_dials.json", "data/new_test_dials.json")


if __name__ == "__main__":
    main()

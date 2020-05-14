# obtain processed dialog data file, where utterance is tokenized, domain-slot become two words, value is also tokenized
import json
from collections import Counter
from pytorch_pretrained_bert import BertTokenizer, BertConfig


SLOTS = ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day', 'hotel-book people', 'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure', 'train-arriveby', 'train-book people', 'train-leaveat', 'attraction-area', 'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'attraction-name', 'restaurant-name', 'attraction-type', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure', 'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arriveby']
    
to_new_SLOTS = {'hotel-pricerange':'hotel-price', 'hotel-type':'hotel-type', 'hotel-parking':'hotel-parking', 'hotel-book stay':'hotel-stay', 'hotel-book day':'hotel-day', 'hotel-book people':'hotel-people', 'hotel-area':'hotel-area', 'hotel-stars':'hotel-stars', 'hotel-internet':'hotel-internet', 'train-destination':'train-destination', 'train-day':'train-day', 'train-departure':'train-departure', 'train-arriveby':'train-arrive', 'train-book people':'train-people', 'train-leaveat':'train-leave', 'attraction-area':'attraction-area', 'restaurant-food':'restaurant-food', 'restaurant-pricerange':'restaurant-price', 'restaurant-area':'restaurant-area', 'attraction-name':'attraction-name', 'restaurant-name':'restaurant-name', 'attraction-type':'attraction-type', 'hotel-name':'hotel-name', 'taxi-leaveat':'taxi-leave', 'taxi-destination':'taxi-destination', 'taxi-departure':'taxi-departure', 'restaurant-book time':'restaurant-time', 'restaurant-book day':'restaurant-day', 'restaurant-book people':'restaurant-people', 'taxi-arriveby':'taxi-arrive'}

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]


def get_new_uttr(l, replaced_to, thresh):
    if len(l) < 1:
        print('empty sub list')
        exit()
    minscore = 100
    start = 0
    for i in range(len(l)- len(replaced_to)+1):
        dis = levenshteinDistance(l[i:i+len(replaced_to)], replaced_to)
        #print(dis/len(replaced_to))
        if dis/len(replaced_to) <= thresh:
            if  dis/len(replaced_to) <  minscore:
                start = i
                minscore = dis/len(replaced_to)
    if minscore < thresh:
        l = l.replace(l[start:start+len(replaced_to)], ' '+replaced_to+' ')

    return l, minscore


def process_file(input_file, output_file):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
    cnt = 0
    cnt_turn = 0
    VALUES = []
    new_dials = {}
    with open(input_file) as f:
        dials = json.load(f)
        for dial_dict in dials:
            new_dial = {}
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_content = {}
                
                turn_uttr = turn["transcript"] + " ; " + turn["system_transcript"] 
                label_dict = {}
                for l in turn["turn_label"]:
                    try:
                        label_dict[ to_new_SLOTS[l[0]].replace('-', ' ') ] = l[1]
                    except:
                        continue
                        
                belief_dict = {}
                for l in turn["belief_state"]:
                    try:
                        belief_dict[ to_new_SLOTS[l['slots'][0][0] ].replace('-', ' ') ] = l['slots'][0][1]
                    except:
                        continue
    
                for slot, value in label_dict.items():
                    cnt_turn += 1
                    if value != '' and value !='none':
                        turn_uttr, score = get_new_uttr(turn_uttr, value, 0.4)
                        turn_uttr = turn_uttr.replace('  ',' ')
                        if value in turn_uttr:
                            cnt += 1
                        else:
                            VALUES.append(value)
                
                new_label_dict = {}         
                for slot, value in label_dict.items():
                    new_label_dict[slot] = tokenizer.tokenize(value)
                
                new_belief_dict = {}         
                for slot, value in belief_dict.items():
                    new_belief_dict[slot] = tokenizer.tokenize(value)
                    
                turn_content['label_dict'] = new_label_dict
                turn_content['belif_dict'] = new_belief_dict
                turn_content['turn_uttr'] = tokenizer.tokenize(turn_uttr)
                        
                        
                new_dial[ turn["turn_idx"] ] = turn_content
                
            new_dials[dial_dict['dialogue_idx']] = new_dial
            
    output = open(output_file, "w")
    json.dump(new_dials, output, indent=4)
    output.close()
    print('finished!')


def main():
    process_file("data/train_dials.json", "data/processed_train_dials.json")
    process_file("data/dev_dials.json", "data/processed_dev_dials.json")
    process_file("data/test_dials.json", "data/processed_test_dials.json")


if __name__ == "__main__":
    main()

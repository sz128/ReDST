import json


def post_process():
    for each_file in ["dev", "test", "train"]:
        #filepath = "./results/turn_belief_pred_%s.json" %(each_file)
        filepath = "./results/turn_belief_pred_lizi_0.48928_%s.json" %(each_file)
        data = json.load(open(filepath))
        for ID, res in data.items():
            last_belief, last_pred_bs_ptr = [], []
            pred_last_bs_score_map = {}
            for turn_id, cont in sorted(res.items(), key=lambda i: int(i[0])):
                cont = convert_label(cont)
                data[ID][turn_id]["turn_belief"] = list(cont["true_labels"])
                data[ID][turn_id]["current_belief"] = list(cont["true_beliefs"])
                data[ID][turn_id]["pred_bs_ptr"] = list(cont["pred_labels"])
                data[ID][turn_id]["last_belief"] = list(last_belief)
                last_belief = list(cont["true_beliefs"])
                data[ID][turn_id]["last_pred_bs_ptr"] = list(last_pred_bs_ptr)
                last_pred_bs_ptr = cont["last_pred_bs_ptr"] if "last_pred_bs_ptr" in cont else {}

                curr_bs_score_map = {}
                for bs in cont["pred_labels"]:
                    slot, value = bs.rsplit("-", 1)
                    curr_bs_score_map[slot] = {"value": value, "score": 1.0}
                data[ID][turn_id]["pred_bs_score_map"] = curr_bs_score_map 

                data[ID][turn_id]["pred_last_bs_score_map"] = dict(pred_last_bs_score_map)
                 
                for slot, value_score in curr_bs_score_map.items():
                    pred_last_bs_score_map[slot] = {
                        "value": value_score["value"],
                        "score": value_score["score"]
                    }
        #outfile_path = "results/turn_belief_pred_processed_%s.json" %(each_file)
        outfile_path = "results/turn_belief_pred_processed_lizi_0.48928_%s.json" %(each_file)
        json.dump(data, open(outfile_path, "w"), indent=4)

   
def convert_label(cont):
    lookup = {'hotel price': 'hotel-pricerange', 'hotel type': 'hotel-type', 'hotel parking': 'hotel-parking', 'hotel stay': 'hotel-book stay', 'hotel day': 'hotel-book day', 'hotel people': 'hotel-book people', 'hotel area': 'hotel-area', 'hotel stars': 'hotel-stars', 'hotel internet': 'hotel-internet', 'train destination': 'train-destination', 'train day': 'train-day', 'train departure': 'train-departure', 'train arrive': 'train-arriveby', 'train people': 'train-book people', 'train leave': 'train-leaveat', 'attraction area': 'attraction-area', 'restaurant food': 'restaurant-food', 'restaurant price': 'restaurant-pricerange', 'restaurant area': 'restaurant-area', 'attraction name': 'attraction-name', 'restaurant name': 'restaurant-name', 'attraction type': 'attraction-type', 'hotel name': 'hotel-name', 'taxi leave': 'taxi-leaveat', 'taxi destination': 'taxi-destination', 'taxi departure': 'taxi-departure', 'restaurant time': 'restaurant-book time', 'restaurant day': 'restaurant-book day', 'restaurant people': 'restaurant-book people', 'taxi arrive': 'taxi-arriveby'}

    result = {}
    for key, res in cont.items():
        tmp_each = []
        for each in res:
            slot, value = each.split("-")
            if slot in lookup:
                slot = lookup[slot]
            tmp_each.append("-".join([slot, value]))
        result[key] = list(tmp_each)
    return result


def main():
    post_process()


if __name__ == "__main__":
    main()

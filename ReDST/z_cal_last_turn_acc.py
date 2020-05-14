import json


def cal_last_acc(which, lastK=1):
    if which == "trade":
        inputfile = "./all_prediction_trade.json"
        print("The result of TRADE is: ")
    if which == "redst":
        inputfile = "./all_prediction_redst_gru_test.json"
        print("The result of ReDST is: ")
    data = json.load(open(inputfile))
    correct, ttl = 0, 0
    for ID, res in data.items():
        for i, (turn_id, cont) in enumerate(sorted(res.items(), key=lambda i: int(i[0]), reverse=True)):
            if i >= lastK:
                break
            ttl += 1
            grd = set(cont["turn_belief"])
            pred = set(cont["pred_bs_ptr"])
            if grd == pred:
                correct += 1
    acc = correct / ttl
    print("Last %d correct/ttl is %d/%d, acc is: %f" %(lastK, correct, ttl, correct/ttl))


def cal_lastK_acc(which, lastK=1):
    if which == "trade":
        inputfile = "./all_prediction_trade.json"
        print("The result of TRADE is: ")
    if which == "redst":
        inputfile = "./all_prediction_redst_gru_test.json"
        print("The result of ReDST is: ")
    data = json.load(open(inputfile))
    correct, ttl = 0, 0
    for ID, res in data.items():
        for i, (turn_id, cont) in enumerate(sorted(res.items(), key=lambda i: int(i[0]), reverse=True)):
            if i != (lastK-1):
                continue
            else:
                ttl += 1
                grd = set(cont["turn_belief"])
                pred = set(cont["pred_bs_ptr"])
                if grd == pred:
                    correct += 1
                break
    acc = correct / ttl
    print("The last %d-th correct/ttl is %d/%d, acc is: %f" %(lastK, correct, ttl, correct/ttl))


def cal_all_acc(which):
    if which == "redst":
        inputfile = "./all_prediction_redst_gru_test.json"
        print("The result of ReDST is: ")
    if which == "noprop":
        inputfile = "./all_prediction_redst_gru_test_noprop.json"
        print("The result of noprop is: ")
    data = json.load(open(inputfile))
    correct, ttl = 0, 0
    for ID, res in data.items():
        for i, (turn_id, cont) in enumerate(sorted(res.items(), key=lambda i: int(i[0]), reverse=True)):
            ttl += 1
            grd = set(cont["turn_belief"])
            pred = set(cont["pred_bs_ptr"])
            if grd == pred:
                correct += 1
    acc = correct / ttl
    print("The overall accuracy is: ", acc)


def cal_slot_error_rate(which):
    if which == "trade":
        inputfile = "./all_prediction_trade.json"
        #print("The slot error rate of TRADE is: ")
    if which == "redst":
        inputfile = "./all_prediction_redst_gru_test.json"
        #print("The slot error rate of ReDST is: ")
    data = json.load(open(inputfile))
    slot_stat = {}
    precision_slot_stat = {}
    ttl_data_points = 0
    for ID, res in data.items():
        ttl_data_points += len(res)
        for i, (turn_id, cont) in enumerate(sorted(res.items(), key=lambda i: int(i[0]), reverse=True)):
            grd = set(cont["turn_belief"])
            pred = set(cont["pred_bs_ptr"])
            for each_grd in grd:
                slot = "-".join(each_grd.split("-")[0:2])
                if slot not in slot_stat:
                    slot_stat[slot] = {"ttl": 1, "hit": 0}
                else:
                    slot_stat[slot]["ttl"] += 1
                if each_grd in pred:
                    slot_stat[slot]["hit"] += 1
            for each_pred in pred:
                slot = "-".join(each_pred.split("-")[0:2])
                if slot not in precision_slot_stat:
                    precision_slot_stat[slot] = {"ttl": 1, "correct": 0, "wrong": 0}
                else:
                    precision_slot_stat[slot]["ttl"] += 1
                if each_pred in grd:
                    precision_slot_stat[slot]["correct"] += 1
                else:
                    precision_slot_stat[slot]["wrong"] += 1
    result_recall = {}
    result_error = {}
    for slot, stat in sorted(slot_stat.items(), key=lambda i: i[0]):
        ttl = stat["ttl"]
        hit = stat["hit"]
        slot_stat[slot]["recall"] = hit/ttl
        slot_stat[slot]["error"] = (ttl-hit)/ttl
        result_recall[slot] = slot_stat[slot]["recall"]
        result_error[slot] = slot_stat[slot]["error"]

    result_prec = {}
    result_error_rate = {}
    prec_ttl, prec_correct = 0, 0
    for slot, stat in sorted(precision_slot_stat.items(), key=lambda i: i[0]):
        ttl = stat["ttl"]
        prec_ttl += ttl
        correct = stat["correct"]
        prec_correct += correct
        result_prec[slot] = correct / ttl 
        result_error_rate[slot] = stat["wrong"] / ttl_data_points
    print(prec_correct/prec_ttl)

    return result_error, result_recall, result_prec, result_error_rate


def f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def main():
    #for K in [1,2,3,4]:
    #    #cal_last_acc("trade", lastK=K)
    #    #cal_last_acc("redst", lastK=K)
    #    cal_lastK_acc("trade", lastK=K)
    #    cal_lastK_acc("redst", lastK=K)

    trade_error, trade_recall, trade_precision, trade_error_rate = cal_slot_error_rate("trade")
    redst_error, redst_recall, redst_precision, redst_error_rate = cal_slot_error_rate("redst")
    #print("TRADE\tTRADE\tTRADE\tTRADE\tReDST\tReDST\tReDST\tReDST\tslot")
    #print("error\trecall\tprec\tF1\terror\trecall\tprec\tF1\tslot")
    #for slot, error in sorted(trade_error.items(), key=lambda i: f1(trade_precision[i[0]], trade_recall[i[0]]), reverse=True):
    #     print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s" \
    #         %(trade_error[slot], trade_recall[slot], trade_precision[slot], f1(trade_precision[slot], trade_recall[slot]), \
    #           redst_error[slot], redst_recall[slot], redst_precision[slot], f1(redst_precision[slot], redst_recall[slot]), slot))
    print("TRADE\tReDST\tslot")
    print("error\terror\tslot")
    #for slot, preci in sorted(trade_precision.items(), key=lambda i: i[1]):
    #     print("%.4f\t%.4f\t%s" %(1-trade_precision[slot], 1-redst_precision[slot], slot))
    for slot, error_rate in sorted(trade_error_rate.items(), key=lambda i: i[1], reverse=True):
         print("%.4f\t%.4f\t%s" %(error_rate, redst_error_rate[slot], slot))


if __name__ == "__main__":
    #main()
    cal_all_acc("redst")
    cal_all_acc("noprop")

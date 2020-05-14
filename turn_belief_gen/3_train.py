import os
import datetime
import numpy as np

import torch
from model import MyBERT
from helper import MultiwozData, flat_accuracy, flat_accuracy1


def main():
    dataset = MultiwozData()
    model = MyBERT.from_pretrained("bert-base-uncased", num_token_labels=6, num_seq_labels=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_grad_norm = 5.0
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=0.00003)
    
    for epoch in range(8):
        print("Epoch: %d" %(epoch))
        tr_loss, seq_loss, token_loss = 0, 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        ts = datetime.datetime.now()
        for step, batch in enumerate(dataset.train_loader):
            model.train(True)
            optimizer.zero_grad()
    
            [input_ids, type_ids, labels, classes, mask] = [each.to(device) for each in batch]
            loss, _token_loss, _seq_loss = model(input_ids, type_ids, mask, labels, classes)
            loss.backward()
            
            tr_loss += loss.item()
            seq_loss += _seq_loss.item()
            token_loss += _token_loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
            
            # print train loss per epoch
            if nb_tr_steps % 1000 == 0:
                tf = datetime.datetime.now()
                print("{}  Train loss: {}, seqloss: {}, tokenloss: {}".format(tf-ts, tr_loss/nb_tr_steps, seq_loss/nb_tr_steps, token_loss/nb_tr_steps), " for {} training batches".format(nb_tr_steps))
                ts = datetime.datetime.now()
                
            #if batch_i == int(len(dataset.train_loader) * 0.25):
            #    evaluate(model, dataset, device, metric_best)
        #save_model(model, float(epoch))
    
        # evaluate    
        model.eval()
        eval_loss, eval_accuracy, eval_accuracy1 = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        token_predictions, seq_predictions, token_true_labels,  seq_true_labels= [], [], [], []
        for batch in dataset.test_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, type_ids, labels, classes, mask = batch
            
            with torch.no_grad():
                tmp_eval_loss, _, _ = model(input_ids, type_ids, mask, labels, classes)
                token_logits, seq_logits = model(input_ids, type_ids, mask)
            token_logits = token_logits.detach().cpu().numpy()
            seq_logits = seq_logits.detach().cpu().numpy()
            
            label_ids = labels.to('cpu').numpy()
            class_ids = classes.to('cpu').numpy()
            
            tmp_eval_accuracy = flat_accuracy(token_logits, label_ids)
            tmp_eval_accuracy1 = flat_accuracy1(seq_logits,  class_ids)
            
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            eval_accuracy1 += tmp_eval_accuracy1
            
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation token Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation seq Accuracy: {}".format(eval_accuracy1/nb_eval_steps))

        directory = 'params/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model, directory + '/model.th')


if __name__ == "__main__":
    main()

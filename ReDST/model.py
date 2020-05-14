import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDST(nn.Module):
    def __init__(self, conf):
        super(GRUDST, self).__init__()
        self.gru = nn.GRU(conf["num_labels"], conf["hidden_size"])
        self.input_hidden_w = nn.Linear(conf["num_labels"], conf["hidden_size"])
        self.output_bs_w = nn.Linear(conf["hidden_size"], conf["num_labels"])
        self.output_slotgate_w = nn.Linear(conf["hidden_size"], conf["num_slots"])

    def forward(self, last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, current_slot_gate):
        turn_belief_grd = turn_belief_grd.unsqueeze(0)
        last_belief_grd = last_belief_grd.unsqueeze(0)

        hidden_input = self.input_hidden_w(last_belief_grd)
        _, hidden_state = self.gru(turn_belief_grd, hidden_input)
        bs_output = torch.sigmoid(self.output_bs_w(hidden_state.squeeze(0)))
        loss_bs = F.binary_cross_entropy(bs_output, current_belief)

        slotgate_output = torch.sigmoid(self.output_slotgate_w(hidden_state.squeeze(0)))
        loss_slot_gate = F.binary_cross_entropy(slotgate_output, current_slot_gate)
        return loss_bs, loss_slot_gate, bs_output, slotgate_output


class RNNDST(nn.Module):
    def __init__(self, conf, adjs=None):
        super(RNNDST, self).__init__()
        self.conf = conf
        if self.conf["use_lbp"] or self.conf["use_lbp_new"] or self.conf["use_lbp_update"]:
            input_size = conf["num_labels"] * 2
        else:
            input_size = conf["num_labels"]

        if conf["rnn_cell"] == "GRU":
            if self.conf["use_lbp_anneal"]:
                self.rnn = nn.GRU(conf["num_labels"]*3, conf["rnn_hidden_size"])
            else:
                self.rnn = nn.GRU(conf["num_labels"], conf["rnn_hidden_size"])

        adj, ent2attr_adj, attr2ent_adj = adjs
        self.adj = torch.FloatTensor(adj).to(conf["device"])
        self.ent2attr_adj = torch.FloatTensor(ent2attr_adj).to(conf["device"])
        self.attr2ent_adj = torch.FloatTensor(attr2ent_adj).to(conf["device"])
        
        self.input_hidden_w = nn.Linear(input_size, conf["rnn_hidden_size"])

        slot_Ws, slot_W1s, slot_cls = [], [], []
        for label_num in conf["num_label_list"]:
            slot_cls.append(nn.Sequential(
                nn.Dropout(p=conf["rnn_dropout"]),
                nn.Linear(conf["rnn_hidden_size"], label_num)
            ))
        self.slot_classifiers = nn.ModuleList(slot_cls)


    def forward(self, last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief, epoch=None):
        turn_belief_grd = turn_belief_grd.unsqueeze(0)
        turn_belief_pred = turn_belief_pred.unsqueeze(0)
        last_belief_grd = last_belief_grd.unsqueeze(0)
        last_belief_pred = last_belief_pred.unsqueeze(0)

        if self.conf["use_which"] == "pred":
            last_belief = last_belief_pred
            turn_belief = turn_belief_pred
        if self.conf["use_which"] == "grd":
            last_belief = last_belief_grd
            turn_belief = turn_belief_grd

        last_belief = self.conf["lbp_alpha"] * last_belief
        last_belief_grd = self.conf["lbp_alpha"] * last_belief_grd

        ori_last_belief = last_belief
        ori_turn_belief = turn_belief

        if self.conf["use_lbp"]:
            first_order_prop = F.normalize(torch.matmul(turn_belief, self.adj), p=2, dim=1)
            second_order_prop = F.normalize(torch.matmul(first_order_prop, self.adj), p=2, dim=1)
            third_order_prop = F.normalize(torch.matmul(second_order_prop, self.adj), p=2, dim=1)
            
            if self.conf["lbp_iters"] == 0:
                prop = turn_belief
            if self.conf["lbp_iters"] == 1:
                prop = turn_belief + first_order_prop 
                #prop = first_order_prop 
            if self.conf["lbp_iters"] == 2:
                prop = turn_belief + first_order_prop + second_order_prop
                #prop = second_order_prop
            if self.conf["lbp_iters"] == 3:
                prop = turn_belief + first_order_prop + second_order_prop + third_order_prop
                #prop = third_order_prop
            last_belief = last_belief + self.conf["lbp_beta"] * prop 
 
            last_belief = torch.cat([last_belief, ori_last_belief], dim=-1) 

        if self.conf["use_lbp_new"]:
            attr_prop = F.normalize(torch.matmul(turn_belief, self.ent2attr_adj), p=2, dim=1)
            ent_prop = F.normalize(torch.matmul(attr_prop, self.attr2ent_adj), p=2, dim=1)
            
            prop = attr_prop + ent_prop
            last_belief = last_belief + self.conf["lbp_beta"] * prop 
            last_belief = torch.cat([last_belief, ori_last_belief], dim=-1) 

        if self.conf["use_lbp_update"]:
            if epoch is not None:
                epoch = epoch.squeeze()
                mu = self.conf["mu"]
                p = mu / (mu + torch.exp(epoch/mu))
                ori_last_belief = last_belief + p * last_belief_grd

            turn_prop = F.normalize(torch.matmul(ori_turn_belief, self.adj), p=2, dim=1)
            turn_prop = turn_prop + ori_turn_belief
            ##last_belief = ori_last_belief + self.conf["prop_beta_turn2last"] * turn_prop 
            last_belief = torch.cat([ori_last_belief, self.conf["prop_beta_turn2last"] * turn_prop], dim=-1)

            last_prop = F.normalize(torch.matmul(ori_last_belief, self.adj), p=2, dim=1)
            turn_belief = ori_turn_belief + self.conf["prop_beta_last2turn"] * last_prop

        if self.conf["use_lbp_anneal"]:
            turn_prop = F.normalize(torch.matmul(ori_turn_belief, self.adj), p=2, dim=1)
            turn_prop = turn_prop + ori_turn_belief
            # prop the turn belief to the last_belief to get I_t
            I_t = torch.cat([ori_last_belief, self.conf["prop_beta_turn2last"] * turn_prop], dim=-1)
            # concate I_t and turn_belief and generate the new turn_belief as rnn input
            turn_belief = torch.cat([ori_turn_belief, I_t], dim=-1)

        hidden_input = self.input_hidden_w(last_belief)
        _, hidden_state = self.rnn(turn_belief, hidden_input)
        hidden_state = hidden_state.squeeze(0)

        losses, outputs = [], []
        for i, each_cls in enumerate(self.slot_classifiers):
            each_out = each_cls(hidden_state)
            outputs.append(each_out)
            each_loss = F.cross_entropy(each_out, current_belief[:, i])
            losses.append(each_loss)

        return losses, outputs


class MLPDST(nn.Module):
    def __init__(self, conf, adj=None):
        super(MLPDST, self).__init__()
        self.conf = conf

        if conf["use_lbp"]:
            input_size = conf["num_labels"] * 3
        else:
            input_size = conf["num_labels"] * 2 

        self.mlp = nn.Sequential( 
            nn.Linear(conf["mlp_hidden_size"], conf["mlp_hidden_size"]),
            nn.BatchNorm1d(conf["mlp_hidden_size"]),
            nn.Dropout(p=conf["mlp_dropout"]),
            nn.ReLU(),
            nn.Linear(conf["mlp_hidden_size"], conf["mlp_hidden_size"]),
            nn.BatchNorm1d(conf["mlp_hidden_size"]),
            nn.Dropout(p=conf["mlp_dropout"]),
            nn.ReLU()
        )

        if conf["use_lbp"]:
            self.enc1 = nn.Sequential( 
                nn.Linear(conf["num_labels"]*2, conf["mlp_hidden_size"]),
                #nn.Dropout(p=conf["mlp_dropout"]),
                nn.Sigmoid(),
            )
            self.enc2 = nn.Linear(conf["num_labels"], conf["mlp_hidden_size"])


        self.adj = torch.FloatTensor(adj).to(conf["device"])

        slot_cls = []
        for label_num in conf["num_label_list"]:
            slot_cls.append(nn.Sequential(
                nn.Linear(conf["mlp_hidden_size"], label_num)
            ))
        self.slot_classifiers = nn.ModuleList(slot_cls)


    def forward(self, last_belief_grd, last_belief_pred, turn_belief_grd, turn_belief_pred, current_belief):
        turn_belief_grd = turn_belief_grd
        turn_belief_pred = turn_belief_pred
        last_belief_grd = last_belief_grd
        last_belief_pred = last_belief_pred

        if self.conf["use_which"] == "pred":
            last_belief = last_belief_pred
            turn_belief = turn_belief_pred
        if self.conf["use_which"] == "grd":
            last_belief = last_belief_grd
            turn_belief = turn_belief_grd

        #last_belief = self.conf["lbp_alpha"] * last_belief

        ori_last_belief = last_belief
        ori_turn_belief = turn_belief

        if self.conf["use_lbp"]:
            first_order_prop = F.normalize(torch.matmul(turn_belief, self.adj), p=2, dim=1)
            second_order_prop = F.normalize(torch.matmul(first_order_prop, self.adj), p=2, dim=1)
            third_order_prop = F.normalize(torch.matmul(second_order_prop, self.adj), p=2, dim=1)

            if self.conf["lbp_iters"] == 0:
                prop = turn_belief
            if self.conf["lbp_iters"] == 1:
                prop = turn_belief + first_order_prop 
                #prop = first_order_prop
            if self.conf["lbp_iters"] == 2:
                prop = turn_belief + first_order_prop + second_order_prop
                #prop = first_order_prop + second_order_prop
            if self.conf["lbp_iters"] == 3:
                prop = turn_belief + first_order_prop + second_order_prop + third_order_prop
                #prop = first_order_prop + second_order_prop + third_order_prop
            #input_ = torch.cat([turn_belief, last_belief, prop], dim=-1)
            input_1 = self.enc1(torch.cat([last_belief, prop], dim=-1))
            input_ = input_1 + self.enc2(turn_belief)
        else:
            input_ = torch.cat([ori_last_belief, ori_turn_belief], dim=-1)

        hidden_state = self.mlp(input_) 

        losses, outputs = [], []
        for i, each_cls in enumerate(self.slot_classifiers):
            each_out = each_cls(hidden_state)
            outputs.append(each_out)
            each_loss = F.cross_entropy(each_out, current_belief[:, i])
            losses.append(each_loss)

        return losses, outputs

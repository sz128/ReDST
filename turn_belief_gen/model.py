import torch
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer


class MyBERT(PreTrainedBertModel):
    def __init__(self, config, num_token_labels=2, num_seq_labels=2):
        super(MyBERT, self).__init__(config)
        self.num_seq_labels = num_seq_labels
        self.num_token_labels = num_token_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tokenclassifier = nn.Linear(config.hidden_size, num_token_labels)
        self.seqclassifier = nn.Linear(config.hidden_size, num_seq_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, token_labels=None, _class=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        #sequence_output = self.dropout(sequence_output)
        layer_indexes = [-1, -2, -3, -4]
        all_layers = []
        for (j, layer_index) in enumerate(layer_indexes):
            all_layers.append( sequence_output[layer_index] )
        sequence_output = torch.mean(torch.stack(all_layers, dim=0), dim=0)
        sequence_output = self.dropout(sequence_output)
        
        #sequence_output = sequence_output.transpose(1, 0) # [batch_size, seq_len, 768]
        
        pooled_output = self.dropout(pooled_output)
        
        token_logits = self.tokenclassifier(sequence_output)
        seq_logits = self.seqclassifier(pooled_output)

        if token_labels is not None:
            ## token loss
            token_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = token_logits.view(-1, self.num_token_labels)[active_loss]
                active_labels = token_labels.view(-1)[active_loss]
                token_loss = token_loss_fct(active_logits, active_labels)
            else:
                token_loss = token_loss_fct(token_logits.view(-1, self.num_token_labels), token_labels.view(-1))
             
            
            ## seq loss
            seq_loss_fct = CrossEntropyLoss()
            seq_loss = seq_loss_fct(seq_logits.view(-1, self.num_seq_labels), _class.view(-1))
            loss = token_loss + 0.1*seq_loss
            return loss,token_loss, seq_loss
        else:
            return token_logits, seq_logits

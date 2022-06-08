import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import load_config
from model.layers import  Encoder,SoftmaxAttention,RNNDropout
from model.utils import get_mask, replace_masked

class EnhancedLSTM(nn.Module):
    def __init__(self,args):
        super(EnhancedLSTM,self).__init__()
        self.args = args
        #Random embedding
        self.Embedding = nn.Embedding(4200,self.args.word_dim)
        self.Encoder = Encoder(nn.LSTM,self.args.word_dim,self.args.hidden_dim,bi=True)
        self.Attention = SoftmaxAttention()

        self.RnnDropout = RNNDropout(p=self.args.dropout)

        self.Projection = nn.Sequential(nn.Linear(4 * 2 * self.args.hidden_dim,
                                                   self.args.hidden_dim),
                                         nn.ReLU())
        self.Composition = Encoder(nn.LSTM,self.args.hidden_dim,self.args.hidden_dim,bi=True)
        self.Classification = nn.Sequential(nn.Dropout(p=self.args.dropout),
                                             nn.Linear(2*4*self.args.hidden_dim,
                                                       self.args.hidden_dim),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.args.dropout),
                                             nn.Linear(self.args.hidden_dim,
                                                       self.args.class_size))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)



    def forward(self,pre_batch,pre_len,hypo_batch,hypo_len):
        pre_mask = get_mask(pre_batch,pre_len).cuda()
        hypo_mask = get_mask(hypo_batch,hypo_len).cuda()

        embedding_pre = self.Embedding(pre_batch)
        embedding_hypo = self.Embedding(hypo_batch)

        embedding_pre = self.RnnDropout(embedding_pre)
        embedding_hypo = self.RnnDropout(embedding_hypo)

        encoded_pre = self.Encoder(embedding_pre,pre_len)
        encoded_hypo = self.Encoder(embedding_hypo, hypo_len)

        attended_pre, attended_hypo =self.Attention(encoded_pre, pre_mask,encoded_hypo, hypo_mask)

        enhanced_pre = torch.cat([encoded_pre,attended_pre,encoded_pre - attended_pre,
                                  encoded_pre * attended_pre],dim=-1)
        enhanced_hypo = torch.cat([encoded_hypo,attended_hypo,encoded_hypo -attended_hypo,
                                   encoded_hypo *attended_hypo],dim=-1)

        projected_pre = self.Projection(enhanced_pre)
        projected_hypo = self.Projection(enhanced_hypo)

        projected_pre = self.RnnDropout(projected_pre)
        projected_hypo = self.RnnDropout(projected_hypo)

        v_ai = self.Composition(projected_pre,pre_len)
        v_bj = self.Composition(projected_hypo,hypo_len)

        v_a_avg = torch.sum(v_ai * pre_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(pre_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypo_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(hypo_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, pre_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypo_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self.Classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits,probabilities


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0


import torch
import torch.nn as nn
from model.utils import masked_softmax,weighted_sum

class RNNDropout(nn.Dropout):
    def forward(self, seq_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = seq_batch.data.new_ones(seq_batch.shape[0],
                                             seq_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * seq_batch

class Encoder(nn.Module):
    def __init__(self,rnn_type,word_dim,hidden_dim,num_layers=1,bias=True,bi=False):
        super(Encoder,self).__init__()
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"
        self.rnn_type = rnn_type
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.encoder = self.rnn_type(self.word_dim,self.hidden_dim,
                                     num_layers=self.num_layers,batch_first=True,bidirectional=bi)
    def forward(self,seq_batch,seq_len):

        packed_batch = nn.utils.rnn.pack_padded_sequence(seq_batch,seq_len.cpu(),
                                                         batch_first=True,enforce_sorted=False)
        packed_output,_ = self.encoder(packed_batch)
        unpacked_output,_ = nn.utils.rnn.pad_packed_sequence(sequence=packed_output, batch_first=True)
        return unpacked_output


class SoftmaxAttention(nn.Module):
    def forward(self,pre_batch,pre_mask,hypo_batch,hypo_mask):
        """
        pre_batch: (batch,pre_len,hidden_dim)
        hypo_batch: (batch,hypo_len,hidden_dim)

        similarity: (batch,pre_len,hypo_len)
        pre_hypo_attn: (batch,pre_len,hypo_len)
        hypo_pre_attn: (batch,hypo_len,pre_len)

        attented_pre: (batch,pre_len,hidden_dim)
        attented_hypo: (batch,hypo_len,hidden_dim)
        """
        similarity = pre_batch.bmm(hypo_batch.transpose(2,1).contiguous())

        pre_hypo_attn = masked_softmax(similarity,hypo_mask)
        hypo_pre_attn = masked_softmax(similarity.transpose(1,2).contiguous(),
                                      pre_mask)


        attented_pre = weighted_sum(hypo_batch,pre_hypo_attn,pre_mask)
        attented_hypo = weighted_sum(pre_batch,hypo_pre_attn,hypo_mask)

        return attented_pre,attented_hypo


if __name__ == "__main__":

    pre = torch.randn((12,265,300))
    hypo = torch.randn((12, 159, 300))
    similarity = pre.bmm(hypo.transpose(2,1))
    pass




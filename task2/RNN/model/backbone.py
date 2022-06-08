import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import load_config

class RNN(nn.Module):
    def __init__(self,args):
        super(RNN,self).__init__()
        self.args = args
        #Random embedding
        self.Embedding = nn.Embedding(15220,self.args.word_dim)
        self.LSTM = nn.LSTM(self.args.word_dim,self.args.hidden_dim,
                            num_layers=self.args.num_layers,batch_first=True)
        #FC with concat features
        self.classifier = nn.Sequential(nn.Dropout(self.args.dropout),
                                        nn.Linear(self.args.hidden_dim,self.args.class_size),
                                        nn.Softmax(dim=1))

    def forward(self,tokens):
        x = self.Embedding(tokens)
        x,_ = self.LSTM(x)
        x = self.classifier(x[:,-1,:])
        return x



if __name__ == '__main__':


    # input = torch.randint(1,10,(128,50))
    #
    #
    # model = RNN(load_config())
    #
    # res = model(input)
    a = [{"pre":1},{"pre":2},{"pre":3}]
    b = a["pre"]

    pass



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import load_config

class CNN(nn.Module):
    def __init__(self,args):
        super(CNN,self).__init__()
        self.args = args

        assert(len(self.args.filters)==len(self.args.filter_num))

        #Random embedding
        self.Embedding = nn.Embedding(15220,self.args.word_dim)
        #Conv1d with different kernal size
        for i in range(len(self.args.filters)):
            temp_conv = nn.Conv1d(in_channels=self.args.word_dim,out_channels=self.args.filter_num[i],
                                  kernel_size=self.args.filters[i])
            setattr(self,f'Conv_{i}',temp_conv)
        #FC with concat features
        self.FC = nn.Linear(sum(self.args.filter_num),self.args.class_size)
        self.softmax = nn.Softmax(dim=1)

    def get_conv(self, i):
        return getattr(self, f'Conv_{i}')

    def forward(self,tokens):
        x = self.Embedding(tokens).permute(0,2,1)
        conv_results = [F.relu(self.get_conv(i)(x)) for i in range(len(self.args.filters))]
        conv_results = [F.max_pool1d(x,x.shape[-1]).view(x.shape[0],-1) for x in conv_results]

        x = torch.cat(conv_results,1)
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.FC(x)
        x = self.softmax(x)
        return x



if __name__ == '__main__':
    # Embedding = nn.Embedding(4000,256)
    # Conv1d = nn.Conv1d(in_channels=256,out_channels=100,kernel_size=2)
    # FC = nn.Linear(100,5)
    #
    # input = torch.randint(1,10,(128,50))
    # eoutput = Embedding(input).permute(0,2,1)
    # coutput = Conv1d(eoutput)
    #
    # Maxpool1d = nn.MaxPool1d(kernel_size=coutput.shape[2])
    # moutput = Maxpool1d(coutput)
    # moutput = moutput.view(moutput.shape[0],-1)
    #
    # foutput = FC(moutput)
    #
    # cc = moutput
    #
    # x=torch.cat([moutput,cc],dim=1)
    model = CNN(load_config())
    inp = torch.randint(0,10,(128,50))
    res = model(inp)


    pass



import argparse
import torch.nn as nn
class_map = {"entailment":0,"neutral":1,"contradiction":2}
def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_mode', type=bool, default=False)

    parser.add_argument('--data_dir', type=str, default=r"../data")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_pre_len', type=int, default=120)
    parser.add_argument('--max_hypo_len', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--class_size', type=int, default=3)

    parser.add_argument('--word_dim', type=int, default=256)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--tag', type=str, default="main")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--rnn_type', default=nn.LSTM)
    parser.add_argument('--max_gradient_norm', type=float, default=10.0)
    parser.add_argument('--class_map', type=dict, default=class_map)
    return parser.parse_args()

if __name__=='__main__':
    arg = load_config()
    assert issubclass(arg.rnn_type, nn.RNNBase), \
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
    print(arg)
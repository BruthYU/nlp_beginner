import argparse

def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_mode', type=bool, default=False)

    parser.add_argument('--data_dir', type=str, default=r"../data")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--class_size', type=int, default=5)
    parser.add_argument('--filters', type=list, default=[3,4,5])
    parser.add_argument('--filter_num', type=list, default=[100, 100, 100])
    parser.add_argument('--word_dim', type=int, default=256)
    parser.add_argument('--dropout',type=float,default=0.4)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--norm_limit', type=float, default=3)
    #parser.add_argument('--weight_decay', type=float, default=5e-4)
    return parser.parse_args()

if __name__=='__main__':
    arg = load_config()
    print(arg.filters)


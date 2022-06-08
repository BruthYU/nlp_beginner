import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import keras.preprocessing.sequence as sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import pickle

def load_json(tag):
    DATA_DIR = "../data/snli_1.0"
    DATA = {'pre':[],'hypo':[],'label':[]}

    with open(os.path.join(DATA_DIR, 'snli_1.0_' + tag + '.jsonl')) as f:
        for line in f.readlines():
            d = json.loads(line)
            if d['gold_label'] != '-':
                DATA['pre'].append(d['sentence1'])
                DATA['hypo'].append(d['sentence2'])
                DATA['label'].append(d['gold_label'])
    dt = pd.DataFrame(DATA)

    return dt

class SimpleTokenizer:
    def __init__(self):
        if not os.path.isfile('tokenizer.pickle'):
            train_data = load_json("main")
            dev_data = load_json("dev")
            text = train_data['pre'] + dev_data['pre']

            tokenizer = Tokenizer(lower=True,num_words=4200,oov_token='<oov>',char_level=False,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
            tokenizer.fit_on_texts(text.fillna(""))
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tokenizer = pickle.load(open('tokenizer.pickle','rb'))
        self.tokenizer = tokenizer
        self.num_words = self.tokenizer.num_words

    def encode(self,text):
        res = self.tokenizer.texts_to_sequences([text])
        return res

class ESIMDataset(Dataset):
    def __init__(self,args):
        self.data = load_json(args.tag)
        self.tokenizer = SimpleTokenizer()
        self.class_size = args.class_size
        self.class_map = args.class_map
        self.max_pre_len = args.max_pre_len
        self.max_hypo_len = args.max_hypo_len

    def fixedLen(self,text,max_len):
        text = text[0][:max_len]
        padding = [0 for _ in range(max_len-len(text))]
        text += padding
        return text

    def __getitem__(self,idx):
        raw_pre,raw_hypo,raw_label = self.data["pre"][idx],\
                                     self.data["hypo"][idx],self.data["label"][idx]
        encode_pre = self.tokenizer.encode(raw_pre)
        encode_hypo = self.tokenizer.encode(raw_hypo)

        pre = torch.tensor(self.fixedLen(encode_pre,self.max_pre_len))
        hypo = torch.tensor(self.fixedLen(encode_hypo,self.max_hypo_len))
        label = torch.tensor(self.class_map[raw_label])

        pre_len = len(raw_pre)

        return {"id":pre,"pre":pre,"pre_len":len(encode_pre[0]),
                "hypo":hypo,"hypo_len":len(encode_hypo[0]),"label":label}

    def __len__(self):
        return self.data.shape[0]

#args.tag main evaluate dev
def load_data(args):
    data = ESIMDataset(args)
    loader = DataLoader(data,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    return data,loader



if __name__ =='__main__':
    s = ["I like","you"]
    lens = [len(seq.split()) for seq in s]
    print(max(lens))

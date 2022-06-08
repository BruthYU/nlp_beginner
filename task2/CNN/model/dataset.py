import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import datatable as dt
from simple_tokenizer import SimpleTokenizer
import torch.nn.functional as F

class SentimentDataset(Dataset):
    def __init__(self,args):
        filename = os.path.join(args.data_dir,"train.tsv")
        file = dt.fread(filename)
        self.data = file.to_pandas()
        self.tokenizer = SimpleTokenizer()
        self.max_len = args.maxlen
        self.class_size = args.class_size

    def __getitem__(self, idx):
        raw_text,raw_label = self.data["Phrase"][idx],int(self.data["Sentiment"][idx])
        text = self.tokenizer.encode(raw_text)

        text = text[0][:self.max_len]
        padding = [0 for _ in range(self.max_len-len(text))]
        text += padding

        text = torch.tensor(text)
        label = torch.tensor(raw_label)
        return text,label

    def __len__(self):
        return self.data.shape[0]

class TestDataset(Dataset):
    def __init__(self,args):
        filename = os.path.join(args.data_dir,"test.tsv")
        file = dt.fread(filename)
        self.data = file.to_pandas()
        self.tokenizer = SimpleTokenizer()
        self.max_len = args.maxlen
        self.class_size = args.class_size

    def __getitem__(self, idx):
        raw_text = self.data["Phrase"][idx]
        text = self.tokenizer.encode(raw_text)

        text = text[0][:self.max_len]
        padding = [0 for _ in range(self.max_len-len(text))]
        text += padding

        text = torch.tensor(text)
        return text

    def __len__(self):
        return self.data.shape[0]

def load_data(args):
    data = SentimentDataset(args)
    loader = DataLoader(data,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    return data,loader

def load_test(args):
    data = TestDataset(args)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return data,loader
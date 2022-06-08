#trained for 25 min, accuracy:0.59483, rank:495/860
import os
import math
import pandas as pd
import torch
import torch.nn as nn
from config import load_config
from dataset import load_data,load_test
import datatable as dt
from backbone import CNN
from tqdm import tqdm
import numpy as np

def _eval():
    args = load_config()
    args.test_mode = True

    filename = "checkpoints/checkpoint_model_best.pth"
    checkpoint = torch.load(filename)
    model = CNN(args)
    if args.cuda:
        model.cuda()
    model.load_state_dict(checkpoint["model_state_dict"])

    test_data, test_loader = load_test(args)

    model.eval()

    res = []
    with torch.no_grad():
        for i, (text) in enumerate(tqdm(test_loader)):
            if args.cuda:
                text = text.cuda()
            prediction = model(text)
            prediction = torch.argmax(prediction,dim=1)
            prediction = prediction.cpu().data.numpy().tolist()
            res.extend(prediction)


    sub_file = dt.fread("../data/sampleSubmission.csv")
    sub_file = sub_file.to_pandas()
    sub_file.Sentiment = res
    sub_file.to_csv("../data/Submission.csv",index=False)
if __name__ =='__main__':
    _eval()

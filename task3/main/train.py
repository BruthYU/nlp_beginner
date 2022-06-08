import sys
sys.path.append("..")

import os
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import train, validate


from model.config import load_config
from model.dataset import load_data
from model.backbone import EnhancedLSTM

def main(args):
    # -------------------- Data loading ------------------- #
    train_data,train_loader = load_data(args)
    args.tag = "dev"
    dev_data, dev_loader = load_data(args)
    # -------------------- Model definition ------------------- #
    model = EnhancedLSTM(args).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)
    # -------------------- Training  ------------------- #
    for epoch in range(args.epochs):
        train(model,train_loader,optimizer,criterion,args.max_gradient_norm)
        epoch_time, epoch_loss, epoch_accuracy = validate(model,
                                                          dev_loader,
                                                          criterion)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        scheduler.step(epoch_accuracy)
        print("Training of epoch {} is finished".format(epoch))
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict()},
                   os.path.join("../checkpoint","{}.pth.tar").format(epoch%10))



if __name__ == "__main__":
    args = load_config()
    main(args)

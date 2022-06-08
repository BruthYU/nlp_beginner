import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from config import load_config
from dataset import load_data
from backbone import RNN
from tqdm import tqdm

def save_checkpoint(model,optimizer,epoch):
    print('Model Saving...')

    model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))

def train(epoch,model,train_loader,optimizer,criterion,args):
    model.train()

    losses, step = 0.,0.
    for i,(text,label) in enumerate(tqdm(train_loader)):
        if args.cuda:
            text,label = text.cuda(),label.cuda()

        prediction = model(text)
        optimizer.zero_grad()
        loss = criterion(prediction,label)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(),args.norm_limit)
        losses += loss.item()
        step += 1
        optimizer.step()
    print("======[Epoch: {}], losses: {}======".format(epoch, losses/step))

def main(args):
    model = RNN(args)
    if args.cuda:
        model.cuda()
    train_data, train_loader = load_data(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    for epoch in range(1,args.epochs+1):
        train(epoch,model,train_loader,optimizer,criterion,args)
        save_checkpoint(model,optimizer,epoch)

if __name__ == '__main__':
    args = load_config()
    main(args)
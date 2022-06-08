import time
import torch
import torch.nn as nn
from tqdm import tqdm
from model.utils import correct_predictions

def train(model,dataloader,optimizer,criterion,max_gradient_norm):
    model.train()

    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        pre,pre_len = batch["pre"].cuda(),batch["pre_len"].cuda()
        hypo, hypo_len = batch["hypo"].cuda(), batch["hypo_len"].cuda()
        labels = batch["label"].cuda()

        optimizer.zero_grad()

        logits, probs = model(pre,pre_len,hypo,hypo_len)
        loss = criterion(logits,labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(),max_gradient_norm)
        optimizer.step()

        correct_preds = correct_predictions(probs,labels)

        description = "loss: {:.4f}, batch_accuracy: {:.4f}".format(loss.item(),correct_preds/pre.shape[0])
        tqdm_batch_iterator.set_description(description)


def validate(model, dataloader, criterion):

    # Switch to evaluate mode.
    model.eval()

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["pre"].cuda()
            premises_lengths = batch["pre_len"].cuda()
            hypotheses = batch["hypo"].cuda()
            hypotheses_lengths = batch["hypo_len"].cuda()
            labels = batch["label"].cuda()

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy



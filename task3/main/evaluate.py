import time
import torch
import torch.nn as nn
from tqdm import tqdm
from model.utils import correct_predictions
from model.config import load_config
from model.dataset import load_data
from model.backbone import EnhancedLSTM

def eval(model, dataloader, criterion):

    # Switch to evaluate mode.
    model.eval()

    test_start = time.time()
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

    test_time = time.time() - test_start
    test_loss = running_loss / len(dataloader)
    test_accuracy = running_accuracy / (len(dataloader.dataset))

    return test_time, test_loss, test_accuracy

if __name__ == "__main__":
    args = load_config()
    args.tag = "test"
    test_data, test_loader = load_data(args)

    model = EnhancedLSTM(args).cuda()
    checkpoint = torch.load("../checkpoint/9.pth.tar")
    model.load_state_dict(checkpoint["model"])

    criterion = nn.CrossEntropyLoss()
    test_time, test_loss, test_accuracy = eval(model,test_loader,criterion)
    print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
          .format(test_time, test_loss, (test_accuracy * 100)))
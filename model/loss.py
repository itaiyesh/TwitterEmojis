import torch.nn.functional as F
import torch.nn as nn
import torch
def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def nlll(y_input, y_target):
    criterion = nn.NLLLoss()
    y_target = torch.squeeze(y_target)
    return criterion(y_input, y_target)

def cross_entropy(y_input, y_target):
    criterion =  nn.CrossEntropyLoss()
    return criterion(y_input, y_target)
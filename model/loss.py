import torch.nn.functional as F
import torch.nn as nn

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def my_loss2(y_input, y_target):
    criterion = nn.L1Loss()
    return criterion(y_input, y_target)

def cross_entropy(y_input, y_target):
    criterion =  nn.CrossEntropyLoss()
    return criterion(y_input, y_target)
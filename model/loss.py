import torch.nn.functional as F
import torch.nn as nn
import torch
def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def my_loss2(y_input, y_target):
    # criterion = nn.L1Loss()
    criterion = nn.NLLLoss()
    y_target = torch.squeeze(y_target)
    # print(y_input)
    # print(y_target)
    # print(y_input.shape)
    # print(y_target.shape)
    # exit(0)
    return criterion(y_input, y_target)

def cross_entropy(y_input, y_target):
    criterion =  nn.CrossEntropyLoss()
    return criterion(y_input, y_target)
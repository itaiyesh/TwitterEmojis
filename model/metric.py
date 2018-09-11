import numpy as np
import torch

def accuracy(y_input, y_target):
    # y_target = torch.squeeze(y_target)
    # print("Input: {}".format(y_input.shape))
    # print("Output: {}".format(y_target.shape))
    y_target = np.squeeze(y_target)

    predicted = [np.argmax(label) for label in y_input]

    # print("Predicted: {} \ny_target: {} \nEqual: {} \nlen: {} \nequal/len: {}".format(
    #    predicted, y_target, (predicted == y_target).sum(), len(y_input),
    #     float(float((predicted == y_target).sum()) / len(y_input))
    # ))
    # print("Accurate: {}".format(float(float((predicted == y_target).sum())/len(y_input))))
    return float((predicted == y_target).sum()) / len(y_target)


def my_metric(y_input, y_target):
    assert len(y_input) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_input)


def my_metric2(y_input, y_target):
    assert len(y_input) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_input) * 2

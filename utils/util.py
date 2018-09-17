import os
import math
import argparse
import torch
import numpy as np
import h5py
import logging
import json
from logger import Logger
logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger()

# model_name = 'LSTM2'
# tweets_path = "datasets/processed/tweets_seq.h5"

tweets_path = None


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


#TODO: Same function in train.py
def parse_config():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    return config, args



# def collate_fn(data):
#     global tweets_path
#     dataset = h5py.File(tweets_path, 'r', libver='latest', swmr=True)
#
#     samples = dataset['features']
#     labels = dataset['labels']
#
#     # TODO: Once we have them as number, read it like this!
#     texts = torch.from_numpy(samples[data[0]:data[-1] + 1]).long()
#     labels = torch.from_numpy(labels[data[0]:data[-1] + 1]).long()
#     # print(texts)
#     return [texts, labels]

# def collate_fn(data, tweets_path):
#
#     print(tweets_path)
#
#     exit(0)
#
#     dataset = h5py.File(tweets_path, 'r', libver='latest', swmr=True)
#
#     samples = dataset['features']
#     labels = dataset['labels']
#
#     # TODO: Once we have them as number, read it like this!
#     texts = samples[data[0]:data[-1] + 1]
#     # print(texts)
#     labels = torch.from_numpy(labels[data[0]:data[-1] + 1]).long()#.cuda()
#     # print(texts)
#     lengths = []
#     #TODO: init with <pad>!!!
#     # look for first argument of pad
#     for i in range(data[0], data[-1]+1):
#         sample = samples[i]
#         l = np.nonzero(sample)[0]#sample.nonzero()[-1]#(sample!=0).argmax(axis=0)#.argmax(sample == 0)
#         if len(l) > 0:
#             #TODO: if setnence has no words this should be an error!
#             l = l[-1]+1
#         else:
#             l = len(sample)
#         # print("Sentence: {} (Len: {})".format(sample, l))
#         l = max(l, 1)
#         # print("Sentence: {} (Len: {})".format(sample, l))
#
#         # print("Max: {}".format(l))
#         lengths.append(l)
#
#     joined = list(zip(lengths,texts,labels))
#     # print("joined: {}".format(joined))
#     joined.sort(key = lambda x: x[0], reverse=True)
#     lengths, texts, labels = zip(*joined)
#
#     texts = torch.from_numpy(np.asarray(texts)).long()#.cuda()
#     lengths = list(lengths)
#     # print(lengths)
#     # print(texts.shape)
#     return [(texts, lengths), labels]
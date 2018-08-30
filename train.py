import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from model.metric import *
# from datasets.processed import *
from datasets import HDF5Dataset
from data_loader import *
from trainer import Trainer
from logger import Logger
import h5py
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='')
# tweets_path = 'C:\\Users\\iyeshuru\\PycharmProjects\\twitter\\tweets_2.h5'
tweets_path = 'datasets/processed/tweets.h5'
vocab = 'datasets/processed/vocab.h5'
word_to_ix = {}

def collate_fn(data):

    dataset = h5py.File(tweets_path, 'r', libver='latest', swmr=True)
    samples = dataset['features']
    labels = dataset['labels']

    #TODO: Once we have them as number, read it like this!
    texts = torch.from_numpy(samples[data[0]:data[-1]])
    labels = torch.from_numpy(labels[data[0]:data[-1]])
    return texts, labels



def main(config, resume):
    train_logger = Logger()

    #TODO: Move to prerocessing
    # vocab_size = 10000
    #
    # logger.info("Building vocab")
    # dataset = h5py.File(tweets_path, 'r', libver='latest', swmr=True)
    # samples = dataset['features']
    # labels = dataset['labels']
    #
    # all_words = []
    # for raw_text in tqdm(samples[:100]):
    #     text = clean(raw_text[0])
    #     for word in text.split():
    #         all_words.append(word)
    #
    # freq = FreqDist(all_words)
    #
    # for word, freq in freq.most_common(vocab_size):
    #     word_to_ix[word] = len(word_to_ix)


    dataset = HDF5Dataset(tweets_path)
    # dataset = HDF5DatasetFromRaw(tweets_path, word_to_ix)

    data_loader = TweetsDataLoader(dataset, config)#, collate_fn=collate_fn)
    valid_data_loader = data_loader.split_validation()

    #TODO: need to shuffle, but keep indexes continuous at training!
    my_data_loader = DataLoader(dataset, batch_size = 32, collate_fn=collate_fn)

    # vocab_path = 'C:\\Users\\iyeshuru\\PycharmProjects\\twitter\\vocab.h5'
    # dataset = h5py.File(vocab_path, 'r', libver='latest', swmr=True)


    model = eval(config['arch'])(dataset.features_len, dataset.num_labels, config)#, config['model'])#(config['model'])
    model.summary()

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    # trainer = Trainer(model, loss, metrics,
    #                   resume=resume,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   train_logger=train_logger)

    #TODO: Split validation/training set
    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=my_data_loader,
                      valid_data_loader=my_data_loader,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':

    # print(torch.cuda.is_available())
    # exit(0)
    logger = logging.getLogger()

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

    main(config, args.resume)

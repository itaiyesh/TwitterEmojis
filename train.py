import os
import json
import logging
import argparse
from datasets import HDF5Dataset
from data_loader import *
from trainer import Trainer
from logger import Logger
import h5py
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from model import  *
from torch.utils.data.sampler import Sampler

logging.basicConfig(level=logging.INFO, format='')
model_name = 'SVM'
tweets_path = "datasets/processed/tweets_bow.h5"  # To be set by config

vocab_path = 'datasets/processed/vocab.h5'
labels_path = 'datasets/processed/labels.h5'
word_to_ix = {}


class RangeSampler(Sampler):
    def __init__(self, range):
        self.range = range

    def __iter__(self):
        return iter(self.range)

    def __len__(self):
        return len(self.range)


def collate_fn(data):
    global tweets_path
    dataset = h5py.File(tweets_path, 'r', libver='latest', swmr=True)

    samples = dataset['features']
    labels = dataset['labels']

    # TODO: Once we have them as number, read it like this!
    texts = torch.from_numpy(samples[data[0]:data[-1] + 1]).long()
    labels = torch.from_numpy(labels[data[0]:data[-1] + 1]).long()
    # print(texts)
    return [texts, labels]


def main(config, resume):

    train_logger = Logger()

    # preprocess(sequence_limit=config['models']['LSTM']['sequence_limit'])

    # TODO: Iterate thru each model?
    #TODO: This does not work
    tweets_path = config['models'][model_name]['dataset']

    dataset = HDF5Dataset(tweets_path)

    # TODO: need to shuffle, but keep indexes continuous at training!
    # TODO: This split by percentage!!
    n = list(range(0, len(dataset)))

    validation_size = int(len(dataset) * config['validation']['validation_split'])
    logging.info("Training set: {}. Validaiton set: {}".format(len(dataset)-validation_size, validation_size))

    data_sampler = RangeSampler(n[:-validation_size])
    validation_sampler = RangeSampler(n[-validation_size:])

    data_sampler = RangeSampler(n[:950000])
    validation_sampler = RangeSampler(n[-50001:])

    data_loader = DataLoader(dataset,
                             sampler=data_sampler,
                             batch_size=config['data_loader']['batch_size'],
                             num_workers=config['data_loader']['num_workers'],
                             collate_fn=collate_fn)

    #TODO: I think the sampler isnt working...
    validation_data_loader = DataLoader(dataset,
                                        sampler=validation_sampler,
                                        batch_size=config['data_loader']['batch_size'],
                                        collate_fn=collate_fn,
                                        num_workers=config['data_loader']['num_workers']
                                        )

    # exit(0)
    vocab_file = h5py.File(vocab_path, 'r', libver='latest', swmr=True)
    labels_file = h5py.File(labels_path, 'r', libver='latest', swmr=True)

    # model = eval(config['arch'])(len(vocab_file), len(labels_file), config)
    model = eval(model_name)(len(vocab_file), len(labels_file), config, config['models'][model_name])
    model.summary()

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    # TODO: Split validation/training set
    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=validation_data_loader,
                      train_logger=train_logger)

    trainer.train()


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

from preprocessing import preprocess

if __name__ == '__main__':

    logger = logging.getLogger()

    config, args = parse_config()

    preprocess(config)

    # main(config, args.resume)

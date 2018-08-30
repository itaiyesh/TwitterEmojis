import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from model.metric import *
from datasets.processed import *
from datasets.raw import *
from data_loader import *
from trainer import Trainer
from logger import Logger
import h5py

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    # TODO: Preprocess HERE
    # take from raw data dir and put in processed

    # data_loader = MnistDataLoader(config)
    # dataset = HDF5Dataset()
    dataset = HDF5DatasetFromRaw('C:\\Users\\iyeshuru\\PycharmProjects\\twitter\\tweets_2.h5')
    data_loader = TweetsDataLoader(dataset, config)

    valid_data_loader = data_loader.split_validation()

    # vocab_path = 'C:\\Users\\iyeshuru\\PycharmProjects\\twitter\\vocab.h5'
    # dataset = h5py.File(vocab_path, 'r', libver='latest', swmr=True)


    model = eval(config['arch'])(dataset.features_len, dataset.num_labels, config)#, config['model'])#(config['model'])
    model.summary()

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
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

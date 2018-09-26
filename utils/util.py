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


def parse_config():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    parser.add_argument('-m', '--models_config', default='config/models_config.json', type=str,
                        help='config file path (default: None)')

    parser.add_argument('-pp', '--pre_config', default='config/preprocessing_config.json', type=str,
                        help='config file path (default: None)')

    parser.add_argument('-t', '--test_config', default='config/test_config.json', type=str,
                        help='config file path (default: None)')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')




    args = parser.parse_args()

    config = None
    models_config = None
    preprocessing_config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']

    elif args.config is not None:
        config = json.load(open(args.config))
        logger.info('loaded {}'.format(args.config))

        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    # assert config is not None

    models_config = json.load(open(args.models_config))
    logger.info('loaded {}'.format(args.models_config))

    preprocessing_config = json.load(open(args.pre_config))
    logger.info('loaded {}'.format(args.pre_config))

    test_config = json.load(open(args.test_config))
    logger.info('loaded {}'.format(args.test_config))

    return config,models_config, preprocessing_config,test_config, args

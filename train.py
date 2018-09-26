import logging
import h5py
from torch.utils.data import DataLoader
from data_loader.data_loaders import *
from torch.utils.data.sampler import Sampler
from datasets import *
from trainer import Trainer, TraditionalTrainer
from prepare_data import *
from model import *
from tester import *
# word_to_ix = {}

class RangeSampler(Sampler):
    def __init__(self, range):
        self.range = range

    def __iter__(self):
        return iter(self.range)

    def __len__(self):
        return len(self.range)

def main(config,models_config,preprocessing_config, args):

    resume = args.resume

    train_logger = Logger()

    model_name = config['model_name']
    model_config = models_config['models'][model_name]

    dataset = HDF5Dataset(model_config)

    n = list(range(0, len(dataset)))
    validation_size = int(len(dataset) * config['validation']['validation_split'])
    logging.info("Training set: {}. Validation set: {}".format(len(dataset)-validation_size, validation_size))

    data_sampler = RangeSampler(n[validation_size:])
    validation_sampler = RangeSampler(n[:validation_size])

    #TODO: Remove
    # data_sampler = RangeSampler(n[:10002])
    # validation_sampler = RangeSampler(n[-1002:])

    data_loader = TweetsDataLoader(dataset,
                             sampler=data_sampler,
                             batch_size=config['data_loader']['batch_size'],
                             num_workers=config['data_loader']['num_workers'],
                                   model_config = model_config)

    #TODO: I think the sampler isnt working...
    validation_data_loader = TweetsDataLoader(dataset,
                                        sampler=validation_sampler,
                                        batch_size=config['data_loader']['batch_size'],
                                        num_workers=config['data_loader']['num_workers'],
                                              model_config = model_config)

    # exit(0)
    vocab_file = h5py.File(preprocessing_config['vocab_file'], 'r', libver='latest', swmr=True)
    labels_file = h5py.File(preprocessing_config['labels_file'], 'r', libver='latest', swmr=True)

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    # For traditional training (i.e. NB) we don't need the full-blown trainer.

    if 'is_traditional' in model_config and model_config['is_traditional']:
        model = eval(model_name)(list(range(len(idx2emoji))), config, model_config, preprocessing_config)

        trainer = TraditionalTrainer(model,
                                     config,
                                     data_loader=data_loader,
                                     valid_data_loader=validation_data_loader,
                                     )
    else:
        model = eval(model_name)(len(vocab_file), len(labels_file), config, model_config, preprocessing_config)

        trainer = Trainer(model, loss, metrics,
                          resume=resume,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=validation_data_loader,
                          train_logger=train_logger)

    model.summary()

    trainer.train()


if __name__ == '__main__':

    config, models_config, preprocessing_config, test_config, args = parse_config()

    # prepare_data(config)

    main(config,models_config,preprocessing_config, args)

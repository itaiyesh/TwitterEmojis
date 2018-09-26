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

word_to_ix = {}

class RangeSampler(Sampler):
    def __init__(self, range):
        self.range = range

    def __iter__(self):
        return iter(self.range)

    def __len__(self):
        return len(self.range)

# config = trained model to test
def main(config, models_config, preprocessing_config, test_config, args):

    train_logger = Logger()

    resume = args.resume

    model_name = config['model_name']
    model_config = models_config['models'][model_name]

    dataset = HDF5Dataset(model_config)

    n = list(range(0, len(dataset)))

    data_sampler = RangeSampler(n)
    validation_sampler = RangeSampler(n)

    data_loader = TweetsDataLoader(dataset,
                                   sampler=data_sampler,
                                   batch_size=config['data_loader']['batch_size'],
                                   num_workers=config['data_loader']['num_workers'],
                                   model_config=model_config)

    # TODO: I think the sampler isnt working...
    validation_data_loader = TweetsDataLoader(dataset,
                                              sampler=validation_sampler,
                                              batch_size=config['data_loader']['batch_size'],
                                              num_workers=config['data_loader']['num_workers'],
                                              model_config=model_config)

    # vocab_file = h5py.File(preprocessing_config['vocab_file'], 'r', libver='latest', swmr=True)
    labels_file = h5py.File(preprocessing_config['labels_file'], 'r', libver='latest', swmr=True)

    # TODO word_to_ix should be inside vocabulary
    vocab_file = preprocessing_config['vocab_file']
    vocab = Vocabulary(vocab_file=vocab_file)
    word_to_ix = vocab.word_to_ix

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    # For traditional training (i.e. NB) we don't need the full-blown trainer.
    if 'is_traditional' in config:
        raise Exception("Traditional trainers are not supported for testing.")

    model = eval(model_name)(len(word_to_ix), len(labels_file), config, models_config['models'][model_name],preprocessing_config)

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=validation_data_loader,
                      train_logger=train_logger)

    model = trainer.model

    model.summary()



    for test_name, test in test_config['test_cases'].items():
        logging.info("Running test: {}".format(test_name))

        test_dataset = TestDataset(model_config, preprocessing_config, test['data'], vocab=vocab)

        # 2nd test -  rigid

        test_sampler = RangeSampler(list(range(0, len(test_dataset))))  # all
        test_loader = TestDataLoader(dataset=test_dataset
                                            , batch_size=config['data_loader']['batch_size'],
                                            sampler=test_sampler,
                                            num_workers=config['data_loader']['num_workers'],
                                            model_config=model_config)

        # Runs 1 time

        Tester(model, eval(test['metric']), metrics, test_config['tester'], data_loader=test_loader).test()


        # 1st test - fine tune
        if 'finetune_model' in model_config:
            finetune_model_name = model_config['finetune_model']
            finetune_model =  eval(finetune_model_name)(len(word_to_ix),
                                               len(labels_file),
                                               config,
                                               models_config['models'][model_name],
                                                        preprocessing_config,
                                               2)
            logging.info("Loading weights from {}".format(resume))
            logging.info("Weights for LSTM are frozen!")
            finetune_model.load_pretrained(model)

            n = list(range(0, len(test_dataset)))

            #TODO: Better sample randomly here!
            train_sampler = RangeSampler(n[:test['finetune_n_train']])
            test_sampler = RangeSampler(n[test['finetune_n_train']:])

            train_loader = TestDataLoader(dataset=test_dataset
                                          , batch_size=8,#config['data_loader']['batch_size'],
                                          sampler=train_sampler,
                                          num_workers=config['data_loader']['num_workers'],
                                          model_config=model_config)
            test_loader = TestDataLoader(dataset=test_dataset
                                         , batch_size=8,#config['data_loader']['batch_size'],
                                         sampler=test_sampler,
                                         num_workers=config['data_loader']['num_workers'],
                                         model_config=model_config)

            # print("Len train loader: {}".format(len(train_loader)))
            # print("Len train test: {}".format(len(test_loader)))

            # pretrained_model = SVMEPretrained(len(word_to_ix), len(labels_file), test_config, test_config['models'][model_name])
            # pretrained_model = biLSTMDOPretrained(len(word_to_ix), len(labels_file), test_config,
            #                                       test_config['models'][model_name])


            finetune_model.summary()

            trainer = Trainer(finetune_model, loss, metrics,
                          resume=False,
                          config=test_config,
                          data_loader=train_loader,
                          valid_data_loader=test_loader,
                          train_logger=train_logger)

            trainer.train()

        # # end
        # exit(0)

        # test_dataset = TestDataset(model_config, config, 'datasets/tests/ss_youtube.pickle', vocab=Vocabulary(debug=True),
        #                            word_to_ix=word_to_ix)
        # test_sampler = RangeSampler(list(range(0, len(test_dataset))))  # all
        # youtube_dataloader = TestDataLoader(dataset=test_dataset
        #                                     , batch_size=config['data_loader']['batch_size'],
        #                                     sampler=test_sampler,
        #                                     num_workers=config['data_loader']['num_workers'],
        #                                     model_config=model_config)
        #
        # test_dataset = TestDataset(model_config, config, 'datasets/tests/ss_twitter.pickle', vocab=Vocabulary(debug=True),
        #                            word_to_ix=word_to_ix)
        # test_sampler = RangeSampler(list(range(0, len(test_dataset))))  # all
        # twitter_dataloader = TestDataLoader(dataset=test_dataset
        #                                     , batch_size=config['data_loader']['batch_size'],
        #                                     sampler=test_sampler,
        #                                     num_workers=config['data_loader']['num_workers'],
        #                                     model_config=model_config)
        #
        # test_dataset = TestDataset(model_config, config, 'datasets/tests/se0714.pickle', vocab=Vocabulary(debug=True),
        #                            word_to_ix=word_to_ix)
        # test_sampler = RangeSampler(list(range(0, len(test_dataset))))  # all
        # se0714_dataloader = TestDataLoader(dataset=test_dataset
        #                                    , batch_size=config['data_loader']['batch_size'],
        #                                    sampler=test_sampler,
        #                                    num_workers=config['data_loader']['num_workers'],
        #                                    model_config=model_config)
        #
        # Tester(model, youtube_acc2, metrics, config, data_loader=youtube_dataloader).test()
        # Tester(model, youtube_acc2, metrics, config, data_loader=twitter_dataloader).test()
        #
        # # Tester(model, se0714_acc, metrics, config, data_loader = se0714_dataloader).test()



if __name__ == '__main__':
    config, models_config, preprocessing_config, test_config, args = parse_config()

    main(config, models_config, preprocessing_config, test_config, args)

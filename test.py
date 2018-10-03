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
def main(config, models_config, pre_config, test_config, args):

    train_logger = Logger()

    resume = args.resume

    model_name = config['model_name']
    model_config = models_config['models'][model_name]

    dataset = HDF5Dataset(model_config)

    n = list(range(0, len(dataset)))

    data_sampler = RangeSampler(n)
    validation_sampler = RangeSampler(n)

    #TODO: just load model
    # data_loader = TweetsDataLoader(dataset,
    #                                sampler=data_sampler,
    #                                batch_size=config['data_loader']['batch_size'],
    #                                num_workers=config['data_loader']['num_workers'],
    #                                model_config=model_config)
    #
    # # TODO: I think the sampler isnt working...
    # validation_data_loader = TweetsDataLoader(dataset,
    #                                           sampler=validation_sampler,
    #                                           batch_size=config['data_loader']['batch_size'],
    #                                           num_workers=config['data_loader']['num_workers'],
    #                                           model_config=model_config)

    # vocab_file = h5py.File(preprocessing_config['vocab_file'], 'r', libver='latest', swmr=True)
    labels_file = h5py.File(pre_config['labels_file'], 'r', libver='latest', swmr=True)

    vocab_file = pre_config['vocab_file']
    vocab = Vocabulary(vocab_file=vocab_file)
    word_to_ix = vocab.word_to_ix

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    # For traditional training (i.e. NB) we don't need the full-blown trainer.
    if 'is_traditional' in config:
        raise Exception("Traditional trainers are not supported for testing.")

    model = eval(model_name)(len(word_to_ix), len(labels_file), config, models_config['models'][model_name], pre_config)

    load_model(model,resume)

    model.cuda()

    model.summary()
    # exit(0)
    # trainer = Trainer(model, loss, metrics,
    #                   resume=resume,
    #                   config=config,
    #                   data_loader=data_loader,
    #                   valid_data_loader=validation_data_loader,
    #                   train_logger=train_logger)
    #
    #
    # model = trainer.model

    model.eval()

    model.summary()

    logging.info("Testing some home made input")

    # test_custom_input(model, vocab, test_config,model_config, pre_config, interactive= True)

    logging.info("Testing known datasets")
    for test_name, test in test_config['test_cases'].items():
        logging.info("Running test: {}".format(test_name))

        test_dataset = TestDataset(model_config, pre_config, test['data'], vocab=vocab)

        # 2nd test -  rigid

        test_sampler = RangeSampler(list(range(0, len(test_dataset))))  # all
        test_loader = TestDataLoader(dataset=test_dataset
                                            , batch_size=config['data_loader']['batch_size'],
                                            sampler=test_sampler,
                                            num_workers=config['data_loader']['num_workers'],
                                            model_config=model_config)

        # Runs 1 time

        # Tester(model, eval(test['metric']), metrics, test_config['tester'], data_loader=test_loader).test()


        # 1st test - fine tune
        if 'finetune_model' in model_config:
            finetune_model_name = model_config['finetune_model']
            finetune_model =  eval(finetune_model_name)(len(word_to_ix),
                                                        len(labels_file),
                                                        config,
                                                        models_config['models'][model_name],
                                                        pre_config,
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


            finetune_model.summary()

            # Remember we've added additional layer, so accuracy is the simple one.
            trainer = Trainer(finetune_model, loss, [accuracy],
                          resume=False,
                          config=test_config,
                          data_loader=train_loader,
                          valid_data_loader=test_loader,
                          train_logger=train_logger)

            trainer.train()


def test_custom_input(model,vocab,test_config,model_config,pre_config,interactive= False):
    # TODO: Remove. this is for debug only
    logging.info("Testing some sentences...")
    texts = ["mmmm....that's very interesting...",
             "This is shit!!",
             "This is the shit!!!",
             "WOW can't believe it!!",
             "hello cutie!!",
             # TODO: Remove these two
             "I love cruising with my homies",
             "I Love you and now you're just gone..."
             ]
    for text in texts:
        logging.info("{}: {}".format(text, predict(model, vocab, test_config,model_config, pre_config, text)))

    if interactive:
        text = ""
        logging.info("Input text, 'bye' to exit.")
        while text != 'bye':
            text = input()
            logging.info(predict(model, vocab, test_config,model_config, pre_config, text))

def predict( model, vocab, test_config, model_config, pre_config, text):
    gpu = torch.device('cuda:' + str(test_config['gpu']))

    is_bow = 'is_bow' in model_config and model_config['is_bow']
    padding = 'pad_input' in model_config and model_config['pad_input']
    #TODO: make configurable
    seq_limit = pre_config['sequence_limit']

    if is_bow:
        seq = make_bow_vector(vocab.clean(text), vocab.word_to_ix)
    else:
        seq = make_seq_vector(vocab.clean(text),  vocab.word_to_ix, seq_limit)
    v = np.repeat([seq], model.batch_size, axis=0)
    if padding:
        f = np.nonzero(seq)[0][-1] + 1
        l = np.repeat(f, model.batch_size, axis=0)
        v = torch.from_numpy(v).long()
        v = v.to(gpu)
        v.requires_grad = False

        output = model((v, l))

    else:
        v = torch.from_numpy(v).long()
        v = v.to(gpu)
        v.requires_grad = False

        output = model(v)

    array = output.cpu().data.numpy()
    array = array[0]
    indices = array.argsort()[-3:][::-1]
    return ", ".join([list(idx2emoji.keys())[i] for i in indices])



if __name__ == '__main__':
    config, models_config, preprocessing_config, test_config, args = parse_config()

    main(config, models_config, preprocessing_config, test_config, args)

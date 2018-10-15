import numpy as np
import torch
from base import BaseTrainer
from tqdm import tqdm
from prepare_data import *
from base import BaseTrainer
from tqdm import tqdm
from prepare_data import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None, vocab_file_path = None, model_config = None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))

        # For debug
        self.vocab = Vocabulary(vocab_file=vocab_file_path, debug=True) if vocab_file_path else None
        self.model_config = model_config
        # print(self.model_config)


        #For graphs
        # TODO: Load from file on resume
        # self.loss_per_iteration_file = os.path.join(self.checkpoint_dir, 'loss_per_iteration.txt')
        # self.iteration_file = os.path.join(self.checkpoint_dir, 'iteration.txt')
        #
        # self.loss_per_iteration_array = []
        # self.iteration_array = []

        self.loss_per_iteration = os.path.join(self.checkpoint_dir, 'loss_per_iteration.pickle')
        self.df = pd.DataFrame(columns=['iteration', 'loss'])

        self.val_loss_per_iteration = os.path.join(self.checkpoint_dir, 'val_loss_per_iteration.pickle')
        self.val_df = pd.DataFrame(columns=['iteration', 'loss'])
    def _to_tensor(self, data, target):
        with_lengths = False
        # Checking whether data is a tuple of data+lengths (for padding)
        if isinstance(data, tuple):
            (data, lengths), target = (torch.LongTensor(data[0]), data[1]), torch.LongTensor(target)
            with_lengths = True
        else:
            data, target = torch.LongTensor(data), torch.LongTensor(target)

        data, target = torch.LongTensor(data), torch.LongTensor(target)
        if self.with_cuda:
            data, target = data.to(self.gpu), target.to(self.gpu)

        if with_lengths:
            return (data, lengths), target
        else:
            return data, target

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        # output = np.argmax(output, axis=1)
        # for i, metric in enumerate(self.metrics):
        #     acc_metrics[i] += metric(output, target)
        # return acc_metrics

        # Specific handling of loss function
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

    # TODO: Remove. this is for debug only
    def _predict(self, text, word_to_ix):
        self.model.on_batch()

        self.model.eval()

        if not self.vocab:
            return

        vocab = self.vocab

        config = self.config

        is_bow = 'is_bow' in self.model_config and self.model_config['is_bow']
        padding = 'pad_input' in self.model_config and self.model_config['pad_input']
        #TODO: make configurable
        seq_limit = 56#self.config['sequence_limit']

        if is_bow:
            seq = make_bow_vector(vocab.clean(text), word_to_ix)
        else:
            seq = make_seq_vector(vocab.clean(text), word_to_ix, seq_limit)
        v = np.repeat([seq], self.model.batch_size, axis=0)
        if padding:
            f = np.nonzero(seq)[0][-1] + 1
            l = np.repeat(f, self.model.batch_size, axis=0)
            v = torch.from_numpy(v).long()
            v = v.to(self.gpu)

            output = self.model((v, l))

        else:
            v = torch.from_numpy(v).long()
            v = v.to(self.gpu)
            output = self.model(v)

        array = output.cpu().data.numpy()
        array = array[0]
        indices = array.argsort()[-3:][::-1]
        self.model.train()
        return ", ".join([list(idx2emoji.values())[i].split("|")[0] for i in indices])

        # return ", ".join([list(idx2emoji.keys())[i] for i in indices])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        # TODO: Remove, debug
        vocab_file = 'datasets/processed/vocab.h5'
        word_to_ix = {}
        with h5py.File(vocab_file, 'r') as h5f:
            for ds in h5f.keys():
                word_to_ix[ds] = int(h5f[ds].value)

        average_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(self.data_loader), total = len(self.data_loader)):
            data, target = self._to_tensor(data, target)
            self.optimizer.zero_grad()

            self.model.batch_size = len(target)
            self.model.on_batch()

            output = self.model(data)


            loss = self.loss(output, target)


            # MY addition #TODO:Remove? it works without
            # loss = torch.autograd.Variable(loss, requires_grad = True)

            # print(loss)
            # print(self.optimizer)
            # if batch_idx == 5:
            #     print("output: {}\nReal: {}\nLoss: {}".format(output, target, loss))
            #     exit(0)
            loss.backward()  # Added retain =true
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            average_loss+=loss.item()

            if batch_idx % 1000 == 0 and batch_idx!=0:
                index = (epoch - 1) * len(self.data_loader) + batch_idx
                self.df.loc[len(self.df)] = [index, average_loss/1000]
                self.df.to_pickle(self.loss_per_iteration)
                average_loss = 0

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
               self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # print(target)

        texts = ["mmmm....that's very interesting...",
                 "This is shit!!",
                 "This is the shit!!!",
                 "WOW can't believe it!!",
                 "hello cutie!!",
                 #TODO: Remove these two
                 "I love cruising with my homies",
                 "I Love you and now you're just gone..."
                 ]

        predictions = {text: self._predict(text, word_to_ix) for text in texts}
        # predictions = {}

        log = {
            # Bug: not all batches are same size (i.e. last one.)
            # This should be a weighted mean instead unless drop_last flag is True on dataloader
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            # Record val loss
            iteration = epoch * len(self.data_loader)
            self.val_df.loc[len(self.val_df)] = [iteration, val_log['val_loss']]
            self.val_df.to_pickle(self.val_loss_per_iteration)
            log = {**log, **val_log, **predictions}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = self._to_tensor(data, target)
                # print(data)

                self.model.batch_size = len(target)
                self.model.on_batch()

                output = self.model(data)
                # print(output)
                # exit(0)

                loss = self.loss(output, target)

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)
                # print(self._eval_metrics(output, target)[-1])

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }


class TraditionalTrainer:

    def __init__(self, model, config, data_loader, valid_data_loader=None, train_logger=None):
        super(TraditionalTrainer, self).__init__()
        self.model = model
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.start_epoch = 1
        self.epochs = config['trainer']['epochs']
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train(epoch)
            self._validate(epoch)

    def _to_numpy(self, data, target):
        return data.cpu().data.numpy().astype(int) ,target.cpu().data.numpy().squeeze().astype(int)

    def _train(self, epoch):
        for batch_idx, (data, target) in tqdm(enumerate(self.data_loader)):
            data, target = self._to_numpy(data, target)
            # print("Data: {}".format(np.argmax(data[0])))
            # print("target: {}".format(target.squeeze().shape))
            self.model.partial_fit(data, target)

            if  batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader)))

    def _validate(self, epoch):
        accuracy = 0
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = self._to_numpy(data, target)

            predicted = self.model.predict(data)

            # precision, recall, fscore, support = score(target, predicted)
            # print('precision: {}'.format(precision))
            # print('recall: {}'.format(recall))
            # print('fscore: {}'.format(fscore))
            # print('support: {}'.format(support))

            accuracy += accuracy_score(target, predicted)


        self.logger.info('accuracy {}'.format(accuracy / len(self.valid_data_loader)))




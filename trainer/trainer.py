import numpy as np
import torch
from base import BaseTrainer
from tqdm import tqdm
from preprocessing import *


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))

    def _to_tensor(self, data, target):
        data, target = torch.LongTensor(data), torch.LongTensor(target)
        if self.with_cuda:
            data, target = data.to(self.gpu), target.to(self.gpu)
        return data, target

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()
        # output = np.argmax(output, axis=1)
        # for i, metric in enumerate(self.metrics):
        #     acc_metrics[i] += metric(output, target)
        # return acc_metrics

        #Specific handling of loss function
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics


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

        #TODO: Remove, debug
        vocab_file = 'datasets/processed/vocab.h5'
        word_to_ix = {}
        with h5py.File(vocab_file, 'r') as h5f:
            for ds in h5f.keys():
                word_to_ix[ds] = int(h5f[ds].value)

        for batch_idx, (data, target) in tqdm(enumerate(self.data_loader)):
            data, target = self._to_tensor(data, target)
            self.optimizer.zero_grad()

            self.model.batch_size = len(target)
            self.model.on_batch()

            output = self.model(data)
            loss = self.loss(output, target)
            # print(loss)
            # print(self.optimizer)
            # if batch_idx == 5:
            #     print("output: {}\nReal: {}\nLoss: {}".format(output, target, loss))
            #     exit(0)
            loss.backward() # Added retain =true
            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # print(target)

        #TODO: Remove this
        is_bow = False
        text1 = "I'm so hungry. this looks yummy!!"
        seq_limit = self.config['sequence_limit']
        if is_bow:
            seq = make_bow_vector(clean(text1), word_to_ix).numpy()
        else:
            seq = make_seq_vector(clean(text1), word_to_ix, seq_limit).numpy()
        v = np.repeat([seq], self.model.batch_size, axis=0)
        v = torch.from_numpy(v).long()  # .cpu()
        v = v.to(self.gpu)
        output = self.model(v)
        array = output.cpu().data.numpy()
        array = array[0]
        indices = array.argsort()[-3:][::-1]
        emotions1 = ", ".join([list(index_to_emotion.keys())[i] for i in indices])

        text2 = "I am so sad. I've been crying all day"
        if is_bow:
            seq = make_bow_vector(clean(text2), word_to_ix).numpy()
        else:
            seq = make_seq_vector(clean(text2), word_to_ix,seq_limit).numpy()
        v = np.repeat([seq], self.model.batch_size, axis=0)
        v = torch.from_numpy(v).long()  # .cpu()
        v = v.to(self.gpu)
        output = self.model(v)
        array = output.cpu().data.numpy()
        array = array[0]
        indices = array.argsort()[-3:][::-1]
        emotions2 = ", ".join([list(index_to_emotion.keys())[i] for i in indices])


        text3 = "mmm...I wonder what I would do..."
        if is_bow:
            seq = make_bow_vector(clean(text3), word_to_ix).numpy()
        else:
            seq = make_seq_vector(clean(text3), word_to_ix,seq_limit).numpy()
        v = np.repeat([seq], self.model.batch_size, axis=0)
        v = torch.from_numpy(v).long()  # .cpu()
        v = v.to(self.gpu)
        output = self.model(v)
        array = output.cpu().data.numpy()
        array = array[0]
        indices = array.argsort()[-3:][::-1]
        emotions3 = ", ".join([list(index_to_emotion.keys())[i] for i in indices])

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist(),
            text1: emotions1,
            text2: emotions2,
            text3: emotions3
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

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

                self.model.batch_size = len(target)
                self.model.on_batch()

                output = self.model(data)
                loss = self.loss(output, target)

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

import numpy as np
import torch
class Tester:

    def __init__(self, model, accuracy, metrics, config, data_loader):
        super(Tester, self).__init__()
        self.config = config
        self.epochs = config['epochs']
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.log_step = int(np.sqrt(self.batch_size))
        self.model = model
        self.accuracy = accuracy
        self.metrics = metrics
        self.start_epoch = 1
        #TODO: modifiable
        self.gpu = 'cuda:0'
        self.with_cuda = True

    def test(self):
        print(self._test())

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = output.cpu().data.numpy()
        target = target.cpu().data.numpy()

        # Specific handling of loss function
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
        return acc_metrics

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

    def _test(self):

        # self.model.eval()
        total_accuracy = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = self._to_tensor(data, target)

                self.model.batch_size = len(target)
                # self.model.on_batch()

                output = self.model(data)
                accuracy = self.accuracy(output, target)

                total_accuracy += accuracy#.item()
                # total_val_metrics += self._eval_metrics(output, target)

        return {
            'test_accuracy': total_accuracy / len(self.data_loader)
            # ,
            # 'val_metrics': (total_val_metrics / len(self.data_loader)).tolist()
        }
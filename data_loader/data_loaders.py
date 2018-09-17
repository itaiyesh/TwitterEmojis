import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.dataloader import default_collate
import h5py
from torch.utils.data import DataLoader

class TweetsDataLoader(DataLoader):
    def __init__(self, dataset,sampler, batch_size, num_workers, model_config):
        super(TweetsDataLoader, self).__init__(dataset = dataset,
                                               sampler = sampler,
                                               batch_size = batch_size,
                                               num_workers = num_workers,
                                               drop_last = True,
                                               collate_fn= self.collate_fn
                                               )

        self.tweets_path = model_config['dataset']
        self.use_padding = model_config['pad_input']

    def collate_pad(self, data, samples, labels):
        # TODO: Once we have them as number, read it like this!
        texts = samples[data[0]:data[-1] + 1]
        # print(texts)
        labels = torch.from_numpy(labels[data[0]:data[-1] + 1]).long()  # .cuda()
        # print(texts)
        lengths = []
        # TODO: init with <pad>!!!
        # look for first argument of pad
        for i in range(data[0], data[-1] + 1):
            sample = samples[i]
            l = np.nonzero(sample)[0]  # sample.nonzero()[-1]#(sample!=0).argmax(axis=0)#.argmax(sample == 0)
            if len(l) > 0:
                # TODO: if setnence has no words this should be an error!
                l = l[-1] + 1
            else:
                l = len(sample)
            # print("Sentence: {} (Len: {})".format(sample, l))
            l = max(l, 1)
            # print("Sentence: {} (Len: {})".format(sample, l))

            # print("Max: {}".format(l))
            lengths.append(l)

        joined = list(zip(lengths, texts, labels))
        # print("joined: {}".format(joined))
        joined.sort(key=lambda x: x[0], reverse=True)
        lengths, texts, labels = zip(*joined)

        texts = torch.from_numpy(np.asarray(texts)).long()  # .cuda()
        lengths = list(lengths)
        # print(lengths)
        # print(texts.shape)
        return [(texts, lengths), labels]

    def collate_no_pad(self, data, samples, labels):
        # TODO: Once we have them as number, read it like this!
        texts = torch.from_numpy(samples[data[0]:data[-1] + 1]).long()
        labels = torch.from_numpy(labels[data[0]:data[-1] + 1]).long()
        # print(texts)
        return [texts, labels]

    def collate_fn(self, data):
        tweets_path = self.tweets_path
        dataset = h5py.File(tweets_path, 'r', libver='latest', swmr=True)
        samples = dataset['features']
        labels = dataset['labels']

        if self.use_padding:
            return self.collate_pad(data, samples, labels)
        else:
            return self.collate_no_pad(data, samples, labels)





from datasets import HDF5Dataset
from prepare_data import *
from model import *
import h5py
from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
from nltk.probability import FreqDist

def main():
    tweets_file = 'datasets/processed/tweets_seq.h5'
    # tweets_file = 'datasets/raw/tweets_seq.h5'
    is_multi_label = False

    tweets = h5py.File(tweets_file, 'r', libver='latest', swmr=True)

    # labels = tweets['labels'][-1600:]
    labels = tweets['labels']#[:16000]
    #
    # sample_indices = list(range(0, len(labels), int(len(labels)/100)))
    # random.shuffle(sample_indices)
    # labels = labels[sample_indices]

    print("Total: {}".format(len(labels)))
    count = []
    multi_class_samples = 0
    for label in tqdm(labels):  # [:10000]:
        if is_multi_label:
            indices =  np.nonzero(label)[0]
            # if len(indices) == 0:
            #     print(label)
            #     exit(0)
            if len(indices) > 1:
                multi_class_samples += 1
            count +=indices.tolist()
        else:
            count.append(label.item())

    print("Multi class samples ratio: {}".format(multi_class_samples/len(labels)))
    # print(count)
    fd = FreqDist(count)
    print(fd.most_common(100))

    plt.hist(count)
    plt.show()



    # This normalizes
    # total = fd.N()

    # for word in fd:
    #     fd[word] /= float(total)

    # fd.plot()
    # print(freq.values())

    # fig, axs = plt.subplots(tight_layout=True)

    # axs.hist(count, bins=25)
    # plt.show()


if __name__ == '__main__':
    main()

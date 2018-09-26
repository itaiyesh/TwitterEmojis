import argparse
import datetime
import logging
import random
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import h5py
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
import utils.index_to_emoji as idx2emoji
from utils.parsing_utils import *
from utils.util import *


from utils import *
import time
from gensim.models import TfidfModel

logging.basicConfig(level=logging.INFO, format='')

logger = logging.getLogger()

script_dir = os.path.dirname(__file__)


# Dataset writing buffer to speed up writing to HDF5 file.
class DatasetBuffer:
    def __init__(self, dataset, buffer_size=2048):
        self.index = 0
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)
        self.count += 1
        if self.count == self.buffer_size:
            self.dataset.resize(self.dataset.shape[0] + self.buffer_size, axis=0)
            self.dataset[self.index:self.index + self.buffer_size] = np.vstack(self.buffer)

            # for i in range(0, self.buffer_size):
            #     self.dataset[self.index] = self.buffer[i]
            #     self.index+=1

            self.count = 0

            self.index += self.buffer_size
            self.buffer = []

    # Adding remaining elements upon file closing
    def close(self):
        if self.count > 0:
            self.dataset.resize(self.dataset.shape[0] + self.count, axis=0)
            self.dataset[self.index:self.index + self.count] = np.vstack(self.buffer)
            self.index += self.count
            self.count = 0
            self.buffer = []


def count_classes(labels):
    logging.info("Counting labels")
    label_count = {}

    for label in tqdm(labels):
        label = np.argmax(label)
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1

    label_count = list(sorted(label_count.items(), key=lambda item: item[1]))
    logging.info(label_count)


def build_vocab(samples, config, sequence_limit_percentile=99.0):

    output_vocab_file = config["vocab_file"]
    tfidf_limit = config["tfidf_limit"]
    w2v_limit = config["w2v_limit"]
    vocab_sample_limit = config['sampling']['vocab_sample_limit'] if 'vocab_sample_limit' in config[
        'sampling'] else None

    # TODO: Remove debug
    return Vocabulary(config, samples, output_vocab_file, tfidf_limit, w2v_limit, vocab_sample_limit, sequence_limit_percentile, logger, debug = True)


def process_samples(config):
    processed_file_bow = config['processed_bow']
    processed_file_sequence = config['processed_sequence']

    output_labels_file = config["labels_file"]

    input_raw_file = config["raw_file"]

    sequence_limit = config["sequence_limit"]
    chunk_lines = config["batch_size"]

    class_limit = config['sampling']['class_limit'] if 'class_limit' in config['sampling'] else None
    total_limit = config['sampling']['total_limit'] if 'total_limit' in config['sampling'] else None

    dataset = h5py.File(input_raw_file, 'r', libver='latest', swmr=True)
    samples = dataset['features']
    labels = dataset['labels']

    if total_limit:
        logging.info("limiting to {} samples".format(total_limit))
        samples = samples[:total_limit]
        labels = labels[:total_limit]

    # count_classes(labels)
    vocab = build_vocab(samples, config)

    logging.info("Writing labels file")
    with h5py.File(output_labels_file, 'w') as h5_bow:
        for i, k in tqdm(enumerate(idx2emoji.keys())):
            h5_bow.create_dataset(k, data=i)

    logger.info("Saved labels file: {} with {} labels ({})".format(output_labels_file, len(idx2emoji),
                                                                   size(os.path.getsize(output_labels_file))))
    logger.info("Preprocessing samples")

    # This enforces an equal sample from each class
    # Note: raw data should already be more or less so
    if class_limit:
        logging.info("Collecting a maximum of {} from each class".format(class_limit))
        class_count = len(idx2emoji) * [class_limit]
        # TODO: Must make sure each class has enough data in samples!
        lines = class_limit * len(idx2emoji)
    else:
        lines = len(samples)

    vocab_size = len(vocab.word_to_ix)

    with h5py.File(processed_file_bow, 'w') as h5_bow, h5py.File(processed_file_sequence, 'w') as h5_seq:
        # Bow
        bow_samples_dataset = h5_bow.create_dataset('features',
                                                    shape=(0, vocab_size),
                                                    maxshape=(lines, vocab_size),

                                                    compression="gzip",
                                                    chunks=(chunk_lines, vocab_size),
                                                    dtype='int32')

        bow_labels_dataset = h5_bow.create_dataset('labels',
                                                   shape=(0, 1),  # labels_n
                                                   maxshape=(lines, 1),

                                                   compression="gzip",
                                                   chunks=(chunk_lines, 1),
                                                   dtype='int32')  # float32 for multi label shit

        # Sequence
        seq_samples_dataset = h5_seq.create_dataset('features',
                                                    shape=(0, sequence_limit),
                                                    maxshape=(lines, sequence_limit),

                                                    compression="gzip",
                                                    chunks=(chunk_lines, sequence_limit),
                                                    dtype='int32')

        seq_labels_dataset = h5_seq.create_dataset('labels',
                                                   shape=(0, 1),  # nlabel for multi label task
                                                   maxshape=(lines, 1),

                                                   compression="gzip",
                                                   chunks=(chunk_lines, 1),
                                                   dtype='int32')

        bow_samples_dataset_buffer = DatasetBuffer(bow_samples_dataset)
        bow_labels_dataset_buffer = DatasetBuffer(bow_labels_dataset)
        seq_samples_dataset_buffer = DatasetBuffer(seq_samples_dataset)
        seq_labels_dataset_buffer = DatasetBuffer(seq_labels_dataset)

        iteration_count = 0
        short_texts_count = 0
        long_texts_count = 0
        satisfied_class_count = 0
        samples_iterate = tqdm(zip(samples, labels), total=len(samples))
        all_classes_satisfied = False
        multi_label_count = 0

        for (sample, _) in samples_iterate:
            iteration_count += 1

            if all_classes_satisfied:
                break

            raw_text = sample[0]
            # TODO: Use clean from previous step
            # TODO: Remove words shorter than 1/2..
            # TODO: Put outside loop!!
            text = vocab.clean(raw_text)  # for some reason, we need this indexing
            text_len = len(text.split())
            if text_len < 2:
                short_texts_count += 1
                continue

            if text_len > sequence_limit:
                long_texts_count += 1
                continue

            # label = make_unique_target_for_maximal_class(raw_text)
            targets = make_targets_for_classes(raw_text)
            if len(targets) > 1:
                multi_label_count += 1

            for label in targets:

                if label is None:
                    # Bad tweet
                    continue

                if class_limit:
                    if class_count[label] == 0:
                        if len(np.nonzero(class_count)[0]) < 1:
                            # Finished Gathering all samples
                            logging.info("All classes have sufficient samples. Breaking")
                            all_classes_satisfied = True
                            break
                        else:
                            # Finished gathering samples for this class
                            continue
                    else:
                        class_count[label] -= 1
                        if class_count[label] == 0:
                            satisfied_class_count += 1
                            if len(idx2emoji) - satisfied_class_count < 4:
                                samples_iterate.set_description(
                                    "{}/{} classes satisfied: Missing {}".format(satisfied_class_count, len(idx2emoji),
                                                                                 ','.join(
                                                                                     [list(idx2emoji.keys())[
                                                                                          class_index]
                                                                                      for class_index in
                                                                                      np.nonzero(class_count)[0]])))
                            else:
                                samples_iterate.set_description(
                                    "{}/{} classes satisfied".format(satisfied_class_count, len(idx2emoji)))

                # pr.enable()

                bow_samples_dataset_buffer.add(make_bow_vector(text, vocab.word_to_ix))
                bow_labels_dataset_buffer.add(label)

                seq_samples_dataset_buffer.add(make_seq_vector(text, vocab.word_to_ix, sequence_limit))
                seq_labels_dataset_buffer.add(label)

                # bow_samples_dataset.resize(bow_samples_dataset.shape[0] + 1, axis=0)
                # bow_labels_dataset.resize(bow_labels_dataset.shape[0] + 1, axis=0)
                #
                # seq_samples_dataset.resize(seq_samples_dataset.shape[0] + 1, axis=0)
                # seq_labels_dataset.resize(seq_labels_dataset.shape[0] + 1, axis=0)
                #
                # bow_samples_dataset[i] = make_bow_vector(text, vocab.word_to_ix)
                # bow_labels_dataset[i] = label
                #
                # seq_samples_dataset[i] = make_seq_vector(text, vocab.word_to_ix, sequence_limit)
                # seq_labels_dataset[i] = label

                # pr.disable()
                # s = io.StringIO()
                # ps = pstats.Stats(pr, stream=s)
                # ps.print_stats()
                # print(s.getvalue())

                # i += 1
                # if i == lines:
                #     break

        samples_iterate.close()

        total_samples = len(seq_samples_dataset)

        bow_samples_dataset_buffer.close()
        bow_labels_dataset_buffer.close()
        seq_samples_dataset_buffer.close()
        seq_labels_dataset_buffer.close()
    logger.info(
        "Of: {} samples iterated, {:.1f}%  had multi-labels. {:.1f}%  were too short. {:.1f}% were too long. Generated {} samples after splitting".format(
            iteration_count,
            (multi_label_count / iteration_count)*100,
            (short_texts_count / iteration_count)*100,
            (long_texts_count / iteration_count)*100,
            total_samples))
    logger.info("Saved bow processed file: {} with {} samples ({})".format(processed_file_bow, lines,
                                                                           size(os.path.getsize(processed_file_bow))))

    logger.info("Saved sequence processed file: {} with {} samples ({})".format(processed_file_sequence, lines,
                                                                                size(os.path.getsize(
                                                                                    processed_file_sequence))))

    if class_limit:
        if len(np.nonzero(class_count)[0]) > 0:
            logging.info(
                "Missing samples for classes {} (total {})".format(np.nonzero(class_count)[0], np.sum(class_count)))


def shuffle(input_file, to_string=True, chunk_lines=32):
    output_file = input_file + 'tmp'

    input_h5 = h5py.File(input_file, 'r', libver='latest', swmr=True)
    n = len(input_h5['features'])
    logging.info("Reading {} lines from dataset".format(n))

    samples = input_h5['features']
    labels = input_h5['labels']

    with h5py.File(output_file, 'w') as output_h5:
        logging.info("Shuffling {}".format(input_file))
        joined = list(zip(samples, labels))
        random.shuffle(joined)
        samples, labels = zip(*joined)

        labels_n = len(labels[0])
        features = len(samples[0])

        dt = h5py.special_dtype(vlen=str) if to_string else 'int32'

        dset1 = output_h5.create_dataset('features',
                                         maxshape=(n, features),
                                         shape=(0, features),
                                         compression="gzip",
                                         chunks=(chunk_lines, features),
                                         dtype=dt)

        dset2 = output_h5.create_dataset('labels',
                                         maxshape=(n, labels_n),
                                         shape=(0, labels_n),
                                         compression="gzip",
                                         chunks=(chunk_lines, labels_n),
                                         dtype='int32')
        logging.info("Writing {}...".format(input_file))
        # TODO: Buffer this writing as well!

        dset1_buffer = DatasetBuffer(dset1)
        dset2_buffer = DatasetBuffer(dset2)

        for i, (sample, label) in tqdm(enumerate(zip(samples, labels)), total=n):
            dset1_buffer.add(sample)
            dset2_buffer.add(label)

        dset1_buffer.close()
        dset2_buffer.close()

        # for i, (sample, label) in tqdm(enumerate(zip(samples, labels)), total=n):
        #     dset1[i] = sample
        #     dset2[i] = label

    input_h5.close()
    os.remove(input_file)
    os.rename(output_file, input_file)

    logger.info("Saved shuffled file: {} with {} samples ({})".format(input_file, n,
                                                                      size(
                                                                          os.path.getsize(input_file))))


def preprocess_mongo(collection, raw_file, limit=None):
    logging.info("Retrieving samples from mongoDB")
    total = limit if limit else collection.estimated_document_count()

    logger.info("Gathering {} documents from mongoDB...".format(total))

    dt = h5py.special_dtype(vlen=str)  # PY3

    cursor = collection.find().limit(total)

    fetch_and_read_batch_size = 1024
    cursor.batch_size(fetch_and_read_batch_size)

    # all_emojis = []
    # for regex in index_to_emotion.values():
    #     emojis = regex.split("|")
    #     for em in emojis:
    #         all_emojis.append(em)
    #
    # all_emojis = "|".join(all_emojis)
    # comp = re.compile(all_emojis)

    bad_samples_count = 0

    with h5py.File(raw_file, 'w') as h5f:
        dset1 = h5f.create_dataset('features',
                                   shape=(total, 1),
                                   compression="gzip",
                                   compression_opts=9,
                                   chunks=(fetch_and_read_batch_size, 1),
                                   dtype=dt)

        dset2 = h5f.create_dataset('labels',
                                   shape=(total, len(idx2emoji)),
                                   compression="gzip",
                                   compression_opts=9,
                                   chunks=(fetch_and_read_batch_size, len(idx2emoji)),
                                   dtype='int32')

        for i, document in tqdm(enumerate(cursor), total=total):
            try:
                raw_text = document['text']

                dset1[i] = raw_text
                label = make_multi_label_target(raw_text)
                if len(np.nonzero(label.numpy())[0]) == 0:
                    bad_samples_count += 1
                dset2[i, :] = make_multi_label_target(raw_text)
            except StopIteration:
                break
            except Exception as e:
                logging.info("{}: Sleeping...".format(e))
                time.sleep(60)

    logging.info("Bad samples count: {}/{}".format(bad_samples_count, total))
    logger.info("Saved raw file: {} with {} samples ({})".format(raw_file, total, size(os.path.getsize(raw_file))))


def preprocess_mongo_balanced_class(collection, output_file, class_limit=None):
    logging.info("Retrieving samples with balance classing from mongoDB")
    min = None
    fetch_multi_label = True
    fetch_and_write_batch = 1024
    dt = h5py.special_dtype(vlen=str)  # PY3

    if not class_limit:
        min_class = None
        logging.info("Counting minimum class")  # This is terribly slow
        for key, value in tqdm(idx2emoji.items(), postfix="Class count"):
            count = collection.count_documents({"text": {'$regex': ".*[{}].*".format(value)}})
            if not min_class or count < min:
                min = count
                min_class = key

            logging.info("Class {} Count {}".format(key, count))

        logging.info("Counted documents. Minimum class {} has {} tweets".format(min_class, min))
    else:
        logging.info("Class limit is {}".format(class_limit))
        min = class_limit

    total = min * len(idx2emoji)

    cursors = {}

    for key, value in idx2emoji.items():
        cursors[key] = collection.find({"text": {'$regex': ".*[{}].*".format(value)}}).limit(min)

    try:
        logging.info("Removing {}".format(output_file))
        os.remove(output_file)
    except:
        logging.info("Could not remove file")

    with h5py.File(output_file, 'w') as h5f:

        dset1 = h5f.create_dataset('features',
                                   shape=(total, 1),
                                   compression="gzip",
                                   compression_opts=9,
                                   chunks=(fetch_and_write_batch, 1),
                                   dtype=dt)

        dset2 = h5f.create_dataset('labels',
                                   shape=(total, len(idx2emoji)),
                                   compression="gzip",
                                   compression_opts=9,
                                   chunks=(fetch_and_write_batch, len(idx2emoji)),
                                   dtype='int32')
        i = 0
        for emotion_index, key in tqdm(enumerate(cursors), postfix="Class Iterate", total=total):
            cursor_name = key
            cursor = cursors[key]
            cursor.batch_size(fetch_and_write_batch)

            logging.info("Fetching {}".format(cursor_name))
            for document in tqdm(cursor, postfix=cursor_name, total=min):
                raw_text = document['text']
                if fetch_multi_label:
                    y = make_multi_label_target(raw_text)
                else:
                    y = make_target_with_unique_class(emotion_index)

                dset1[i] = raw_text
                dset2[i, :] = y

                i += 1

    logger.info(
        "Saved raw file: {} with {} samples ({})".format(output_file, total, size(os.path.getsize(output_file))))


def collect_data(config):
    client = MongoClient(config["mongo"]["ip"], config["mongo"]["port"])
    collection = client[config["mongo"]["db"]]["aug7"]
    output_file = config["raw_file"]

    if 'balanced' in config['sampling']:
        class_limit = config['sampling']['class_limit'] if 'class_limit' in config['sampling'] else None
        preprocess_mongo_balanced_class(collection, output_file, class_limit=class_limit)
    else:
        sample_limit = config['sampling']['sample_limit'] if 'sample_limit' in config['sampling'] else None
        preprocess_mongo(collection, output_file, limit=sample_limit)


def prepare_data(config):
    if config['sampling']['collect_data']:
        collect_data(config)

    process_samples(config)

    processed_file_sequence = config['processed_sequence']
    processed_file_bow = config['processed_bow']

    shuffle(processed_file_sequence, to_string=False)
    shuffle(processed_file_bow, to_string=False)


if __name__ == '__main__':
    logger = logging.getLogger()

    pre_config = parse_config()[2]

    prepare_data(pre_config)

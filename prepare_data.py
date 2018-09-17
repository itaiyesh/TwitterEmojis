import argparse
import datetime
import logging
import random
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import h5py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import utils.index_to_emoji as idx2emoji
from utils.parsing_utils import *
from utils.util import *
from nltk import FreqDist
from utils import *
import time

logging.basicConfig(level=logging.INFO, format='')

logger = logging.getLogger()

script_dir = os.path.dirname(__file__)


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


def build_vocab(samples, config):
    output_vocab_file = config["vocab_file"]
    min_df, max_df = config["sampling"]["vocab_min_df"], config["sampling"]["vocab_max_df"]
    vocab_size = config["tfidf_limit"]
    vocab_sample_limit = config['sampling']['vocab_sample_limit'] if 'vocab_sample_limit' in config[
        'sampling'] else None

    #TODO: Remove debug
    vocab = Vocabulary(debug = True)

    logging.info("Building vocabulary")

    all_sentences = []
    if vocab_sample_limit:
        logging.info("Limiting vocab build up to {} samples".format(vocab_sample_limit))
        vocab_samples = samples[:vocab_sample_limit]
    else:
        vocab_samples = samples

    for sample in tqdm(vocab_samples):
        all_sentences.append(vocab.clean(sample[0]))  # for some reason, we need this indexing

    logging.info("Running TF-IDF on collected words...")
    tvec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=(1, 1), lowercase=False)  # stop_words='english',
    tvec_weights = tvec.fit_transform(all_sentences)
    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
    weights_df.sort_values(by='weight', ascending=False).head(vocab_size)
    logging.info("Chosen {} words for vocab".format(len(weights_df)))
    print(weights_df)
    tfidf_words = [word[0] for word in weights_df.get_values()]

    # Adding most frequent k as suggested by
    # https://openreview.net/pdf?id=Bk8N0RLxx
    all_words =[]
    for sentence in all_sentences:
        for word in sentence.split():
            all_words.append(word)
    # This is using simple frequency
    freq = FreqDist(all_words)
    most_frequent_words = [w[0] for w in freq.most_common(config['frequent_words_limit'])]
    print(most_frequent_words)

    word_to_ix = {}

    # It's important to keep pad first!
    best_words = ['<pad>'] + list(set(tfidf_words).union(set(most_frequent_words))) + ['<unk>']

    for word in best_words:
        word_to_ix[word] = len(word_to_ix)


    try:
        logging.info("Removing {}".format(output_vocab_file))
        os.remove(output_vocab_file)
    except:
        logging.info("Could not remove file")

    logging.info("Writing vocab file")

    with h5py.File(output_vocab_file, 'w') as h5_bow:
        for k, v in tqdm(word_to_ix.items()):
            h5_bow.create_dataset(k, data=v)

    logger.info("Saved vocab file: {} with {} words ({})".format(output_vocab_file, len(word_to_ix),
                                                                 size(os.path.getsize(output_vocab_file))))
    vocab.word_to_ix = word_to_ix
    return vocab


def process_samples(config):
    processed_file_bow = config['processed_bow']
    processed_file_sequence = config['processed_sequence']

    output_labels_file = config["labels_file"]

    input_raw_file = config["raw_file"]

    sequence_limit = config["sequence_limit"]
    chunk_lines = config["data_loader"]["batch_size"]

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
        logging.info("Collecting maximum {} of each class".format(class_limit))
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

        i = 0
        satisfied_class_count = 0
        samples_iterate = tqdm(zip(samples, labels), total=len(samples))

        for (sample, _) in samples_iterate:

            raw_text = sample[0]
            label = 1
            # label = make_unique_target_for_maximal_class(raw_text)
            for label in make_targets_for_classes(raw_text):

                if label is None:
                    # Bad tweet
                    continue

                if class_limit:
                    if class_count[label] == 0:
                        if len(np.nonzero(class_count)[0]) < 1:
                            # Finished Gathering all samples
                            logging.info("All classes have sufficient samples. Breaking")
                            break
                        else:
                            # Finished gathering samples for this class
                            continue
                    else:
                        class_count[label] -= 1
                        if class_count[label] == 0:
                            satisfied_class_count += 1
                            samples_iterate.set_description(
                                "{}/{} classes satisfied".format(satisfied_class_count, len(idx2emoji)))

                # TODO: Use clean from previous step
                text = vocab.clean(raw_text)  # for some reason, we need this indexing

                bow_samples_dataset.resize(bow_samples_dataset.shape[0] + 1, axis=0)
                bow_labels_dataset.resize(bow_labels_dataset.shape[0] + 1, axis=0)

                seq_samples_dataset.resize(seq_samples_dataset.shape[0] + 1, axis=0)
                seq_labels_dataset.resize(seq_labels_dataset.shape[0] + 1, axis=0)

                pr.enable()
                bow_samples_dataset[i] = make_bow_vector(text, vocab.word_to_ix)
                bow_labels_dataset[i] = label

                seq_samples_dataset[i] = make_seq_vector(text, vocab.word_to_ix, sequence_limit)
                seq_labels_dataset[i] = label

                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s)
                ps.print_stats()
                print(s.getvalue())

                i += 1
                if i == lines:
                    break

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
        logging.info("Shuffling {}. This will take approximately {}".format(input_file,
                                                                            datetime.timedelta(
                                                                                seconds=((0.0001175470 * n)))))
        joined = list(zip(samples, labels))
        random.shuffle(joined)
        samples, labels = zip(*joined)

        labels_n = len(labels[0])
        features = len(samples[0])

        dt = h5py.special_dtype(vlen=str) if to_string else 'int32'

        dset1 = output_h5.create_dataset('features',
                                         shape=(n, features),
                                         compression="gzip",
                                         chunks=(chunk_lines, features),
                                         dtype=dt)

        dset2 = output_h5.create_dataset('labels',
                                         shape=(n, labels_n),
                                         compression="gzip",
                                         chunks=(chunk_lines, labels_n),
                                         dtype='int32')
        logging.info("Writing {}...".format(input_file))
        for i, (sample, label) in tqdm(enumerate(zip(samples, labels)), total=n):
            dset1[i] = sample
            dset2[i] = label

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

    config, args = parse_config()
    prepare_data(config)

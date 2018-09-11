from collections import Counter
from nltk.corpus import stopwords, words
from nltk.stem.snowball import SnowballStemmer
import json, pandas, re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
import emoji
import re
from bs4 import BeautifulSoup
# from utils import contraction, correction, stop_words, stemmer, split_hashtag
import itertools
import numpy as np
import h5py
from tqdm import tqdm
import logging
from nltk.probability import FreqDist
import random
import os
import math
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import argparse

logging.basicConfig(level=logging.INFO, format='')

logger = logging.getLogger()

script_dir = os.path.dirname(__file__)


def size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# stemming_____________________________________________________________________________________________________________
stemmer = SnowballStemmer('english')

# contraction__________________________________________________________________________________________________________
# contraction = json.load(open('datasets/constants/contraction.json'))
contraction = json.load(open(os.path.join(script_dir, 'constants/contraction.json')))

# stopword_____________________________________________________________________________________________________________
stop_words = stopwords.words('english')
stop_words.remove('not')


# correction__________________________________________________________________________________________________________
def all_words(text):
    return re.findall('\\w+', text.lower())


WORDS = Counter(all_words(open(os.path.join(script_dir, 'constants/big.txt')).read()))


def P(word, N=sum(WORDS.values())):
    """Probability of `word`."""
    return WORDS[word] / N


def correction(word):
    """Most probable spelling correction for word."""
    return max(candidates(word), key=P)


def candidates(word):
    """Generate possible spelling corrections for word."""
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words):
    """The subset of `words` that appear in the dictionary of WORDS."""
    return set((w for w in words if w in WORDS))


def edits1(word):
    """All edits that are one edit away from `word`."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """All edits that are two edits away from `word`."""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# word dictionary______________________________________________________________________________________________________
word_dictionary = list(set(words.words()))
for alphabet in 'bcdefghjklmnopqrstuvwxyz':
    word_dictionary.remove(alphabet)

useless_two_letter_words = pandas.read_csv(os.path.join(script_dir, 'constants/useless_two_letter_words.csv'))
for word in useless_two_letter_words:
    word_dictionary.remove(word)

useful_words = pandas.read_csv(os.path.join(script_dir, 'constants/useful_words.csv'))
for word in useful_words:
    word_dictionary.append(word)

for key in contraction:
    word_dictionary.append(key)


# split hashtags_______________________________________________________________________________________________________
def split_hashtag(hashtag):
    found = False
    for i in reversed(range(1, len(hashtag) + 1)):
        if stemmer.stem(hashtag[:i]) in word_dictionary or hashtag[:i] in word_dictionary:
            found = True
            if i == len(hashtag):
                if hashtag[:i] in contraction:
                    contraction[hashtag[:i]]
                else:
                    return hashtag[:i]
            else:
                child = split_hashtag(hashtag[i:])
                if child:
                    return hashtag[:i] + ' ' + child
    if not found:
        return False


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence.split():
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec
    # return vec.view(1, -1)


def make_seq_vector(sentence, word_to_ix, sequence_limit):
    sequence = torch.LongTensor(np.zeros(sequence_limit, dtype=np.int64))
    count = 0
    # TODO: need to replace missing words by <em> or something...
    for word in sentence.split():
        if count > sequence_limit - 1:
            break

        if word not in word_to_ix:
            word = '<unk>'

        sequence[count] = word_to_ix[word]
        count += 1

    return sequence


def extract_emojis(str):
    return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

index_to_emotion = {
    'smiling': "😁|😃|😄|🙂|🙃|😸",
    'laughing': "😂|🤣|😅|😆|😹",
    # 'angel': "😇",
    'winking': "😉",
    'blushing': "😊|☺",
    # 'liplicking': "😋", #5
    'calm_smiling': "😌",
    'inlove': "😍|😻|💖|❤",
    'kissing': "😘|😗|😙|😚|😽",
    'playful_tongue_out': "😜|😝|😛",
    # 'puking_money': "🤑", #10  # 186000 ~
    'sun_glasses': "😎",
    # 'nerd_glasses': "🤓",  # 240000~
    'excited_hands_shaking': "🤗",
    # 'sneeky_face': "😏|😼",
    'straight_face': "😐|😑|😒|🙄", #15
    'thinking': "🤔",
    # 'hassing' :"\\U0001f92b",
    # 'lying' : "🤥",#55000~
    'pale_face': "😳|😤",
    'sad_face': "😞|😟|😔|😕|🙁|☹|😣|😖|😫|😩|😰|😦|😧|🙀", #20
    'crying_face': "😢|😥|😪|😭|😿",
    # 'mad_face': "😠|😡|😾|🤬",   # Rem
    'shock_face': "😮|😱|😨|😯|😲",
    # 'hungry_face': "🤤",  # ~467837
    # 'sweat_drop_face': "😓", #25
    # 'fucked_up_face': "😵|\\U0001f92a",
    # 'zip_face': '🤐',
    # 'wounded_face': '🤕',
    # 'doctor_face': '😷',
    # 'sick_face': '🤒|🤧|🤮', #30 # Remove
    # 'vomit_face': '🤢',
    # 'sleepy': '😴|💤',   # Remove
    # 'devil_face': '😈|👿|👹' ,
    # 'exhausted': "😫"
    'eyebrow': "🤨",
    # 'stary_eyes': "🤩",
    # 'laughing_not_sure': "🤭"

}


# index_to_emotion = {
#     'smiling': "😁|😃|😄|🙂|🙃|😸",
#     'laughing': "😂|🤣|😅|😆|😹",
#     'angel': "😇",
#     'winking': "😉",
#     'blushing': "😊|☺",
#     'liplicking': "😋", #5
#     'calm_smiling': "😌",
#     'inlove': "😍|😻|💖|❤",
#     'kissing': "😘|😗|😙|😚|😽",
#     'playful_tongue_out': "😜|😝|😛",
#     'puking_money': "🤑", #10  # 186000 ~
#     'sun_glasses': "😎",
#     'nerd_glasses': "🤓",  # 240000~
#     'excited_hands_shaking': "🤗",
#     'sneeky_face': "😏|😼",
#     'straight_face': "😐|😑|😒|🙄", #15
#     'thinking': "🤔",
#     'hassing' :"\\U0001f92b",
#     'lying' : "🤥",#55000~
#     'pale_face': "😳|😤",
#     'sad_face': "😞|😟|😔|😕|🙁|☹|😣|😖|😫|😩|😰|😦|😧|🙀", #20
#     'crying_face': "😢|😥|😪|😭|😿",
#     'mad_face': "😠|😡|😾|🤬",   # Rem
#     'shock_face': "😮|😱|😨|😯|😲",
#     'hungry_face': "🤤",  # ~467837
#     'sweat_drop_face': "😓", #25
#     'fucked_up_face': "😵|\\U0001f92a",
#     'zip_face': '🤐',
#     'wounded_face': '🤕',
#     'doctor_face': '😷',
#     'sick_face': '🤒|🤧|🤮', #30 # Remove
#     'vomit_face': '🤢',
#     'sleepy': '😴|💤',   # Remove
#     'devil_face': '😈|👿|👹' ,
#     'exhausted': "😫",
#     'eyebrow': "🤨",
#     'stary_eyes': "🤩",
#     'laughing_not_sure': "🤭"
# }

def count_occurrences(text, reg):
    comp = re.compile(reg)
    return len(re.findall(comp, text))


# Creates a multi label, where each class is given 1 if it occurs in text, 0 otherwise
def make_multi_label_target(raw_text):
    vec = torch.LongTensor([(1 if count_occurrences(raw_text, reg) > 0 else 0) for reg in index_to_emotion.values()])
    # vec = F.normalize(vec, p=2, dim=0)
    return vec


# Creates a multi label, where each class is given a proportion of its occurrences in the text
def make_multi_label_proportional_target(raw_text):
    vec = torch.FloatTensor([count_occurrences(raw_text, reg) for reg in index_to_emotion.values()])
    vec = F.normalize(vec, p=2, dim=0)
    return vec


# Creates a label for a given class (index of index_to_emotion)
def make_target_with_unique_class(index):
    vec = np.zeros(len(index_to_emotion))
    vec[index] = 1

    return vec


# Creates a label for the maximal class
def make_unique_target_for_maximal_class(raw_text):
    count_list = [count_occurrences(raw_text, reg) for reg in index_to_emotion.values()]
    count_array = np.array(count_list)
    return -1 if len(np.nonzero(count_array)[0]) < 1 else np.argmax(count_array)


# Creates a label for the maximal class
def make_target(raw_text):
    count_array = np.array([count_occurrences(raw_text, reg) for reg in index_to_emotion.values()])
    vec = np.zeros(len(index_to_emotion))
    vec[np.argmax(count_array)] = 1
    return vec


def clean(tweet_text):
    # ignore non ascii characters
    tweet_text = tweet_text.encode('ascii', 'ignore')
    tweet_text = tweet_text.lower()

    # remove html tags
    tweet_text = BeautifulSoup(tweet_text, 'html.parser').get_text()

    # remove @ references
    tweet_text = re.sub(r'@\w+', ' ', tweet_text)

    # remove 'RT' text
    tweet_text = re.sub(r'(^| )rt ', ' ', tweet_text)

    # remove links
    tweet_text = re.sub(r'https?:\/\/\S*', ' ', tweet_text)

    # Voteeeeeeeee -> Votee
    tweet_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet_text))

    # TODO: This part takes forever!!
    # #split hashtags
    # useless_hashtag = ['tcot', 'tlot', 'ucot', 'p2b', 'p2', 'ccot']
    # split_tweet_text = ''
    # for word in tweet_text.split():
    #     if word.startswith('#'):
    #         split_words = split_hashtag(word[1:]) if word[1:] not in useless_hashtag else ' '
    #         if split_words:
    #             split_tweet_text += (' ' + correction(word[1:]) + ' ' + split_words)
    #         else:
    #             split_tweet_text += (' ' + correction(word[1:]))
    #     else:
    #         #expand contractions
    #         if word in contraction:
    #             split_tweet_text += ' ' + contraction[word]
    #         else:
    #             split_tweet_text += ' ' + correction(word)
    # tweet_text = split_tweet_text

    # remove special char (except #) and contraction for hash tag
    tweet_text = re.sub('[^0-9a-zA-Z]', ' ', tweet_text)

    # TODO: These stuff are not so good for LSTM
    # tokenize
    # TODO: Whats this?
    # words = word_tokenize(tweet_text)

    # remove stopwords
    # words = [stemmer.stem(w) for w in words if not w in stop_words]

    # join the words
    # tweet_text = " ".join(words)

    return tweet_text


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
    print(label_count)
    # print(i[0] for i in label_count)


# TODO: Sequence limit needs to arrive from config!!
def preprocess_hdf5(input_raw_file,
                    processed_file_bow,
                    processed_file_sequence,
                    output_vocab_file,
                    output_labels_file,
                    vocab_size,
                    sequence_limit,
                    chunk_lines,
                    total_limit=None,
                    vocab_sample_limit = None,
                    class_limit=None):
    dataset = h5py.File(input_raw_file, 'r', libver='latest', swmr=True)
    # Note, this magic number is an approximate multiplier for the number of lines
    samples = dataset['features']
    labels = dataset['labels']

    if total_limit:
        logging.info("limiting to {} samples".format(total_limit))
        samples = samples[:total_limit]
        labels = labels[:total_limit]

    # count_classes(labels)

    # else:
    #     samples = samples_
    #     labels = labels_

    word_to_ix = {}

    logging.info("Building vocabulary")

    all_sentences = []
    if vocab_sample_limit:
        logging.info("Limiting vocab build up to {} samples".format(vocab_sample_limit))
        vocab_samples = samples[:vocab_sample_limit]
    else:
        vocab_samples = samples

    for sample in tqdm(vocab_samples):
        all_sentences.append(clean(sample[0]))  # for some reason, we need this indexing

    print(len(all_sentences))
    tvec = TfidfVectorizer(min_df=.0002, max_df=.9, ngram_range=(1, 1))  # stop_words='english',
    tvec_weights = tvec.fit_transform(all_sentences)
    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
    weights_df.sort_values(by='weight', ascending=False).head(vocab_size)
    print(weights_df)

    best_words = ['<unk>'] + [word[0] for word in weights_df.get_values()]
    for word in best_words:
        word_to_ix[word] = len(word_to_ix)

    # all_words =[]
    # for sentence in all_sentences:
    #     for word in sentence.split():
    #         all_words.append(word)
    # # This is using simple frequency
    # freq = FreqDist(all_words)
    # for word, freq in freq.most_common(vocab_size):
    #     word_to_ix[word] = len(word_to_ix)

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

    logging.info("Writing labels file")
    with h5py.File(output_labels_file, 'w') as h5_bow:
        for i, k in tqdm(enumerate(index_to_emotion.keys())):
            h5_bow.create_dataset(k, data=i)

    logger.info("Saved labels file: {} with {} labels ({})".format(output_labels_file, len(index_to_emotion),
                                                                   size(os.path.getsize(output_labels_file))))
    logger.info("Preprocessing samples")

    # This enforces an equal sample from each class
    # Note: raw data should already be more or less so
    if class_limit:
        logging.info("limiting to {} per class".format(class_limit))
        class_count = len(index_to_emotion) * [class_limit]
        # TODO: Must make sure each class has enough data in samples!
        lines = class_limit * len(index_to_emotion)
    else:
        lines = len(samples)

    vocab_size = len(word_to_ix)
    labels_n = len(index_to_emotion)

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
        # TODO: If not all classes satisfy the count, default '0' will fill the remaining array,
        # these count as class 0, which is not good! need to discard tail!
        for (sample, label) in tqdm(zip(samples, labels), total=len(samples)):

            raw_text = sample[0]

            label = make_unique_target_for_maximal_class(raw_text)

            if label == -1:
                continue
                # Bad tweet

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

            # TODO: Use clean from previous step
            text = clean(raw_text)  # for some reason, we need this indexing

            bow_samples_dataset.resize(bow_samples_dataset.shape[0] + 1, axis=0)
            bow_labels_dataset.resize(bow_labels_dataset.shape[0] + 1, axis=0)

            seq_samples_dataset.resize(seq_samples_dataset.shape[0] + 1, axis=0)
            seq_labels_dataset.resize(seq_labels_dataset.shape[0] + 1, axis=0)
            # TODO: Note, we dont even use the label in raw dataset! as we havent decided on multi-label or not yet
            bow_samples_dataset[i] = make_bow_vector(text, word_to_ix)
            bow_labels_dataset[i] = label  # label#make_target(sample[0])

            # Sequence data set
            # label = torch.LongTensor([make_unique_target_for_maximal_class(raw_text)])

            seq_samples_dataset[i] = make_seq_vector(text, word_to_ix, sequence_limit)
            seq_labels_dataset[i] = label

            # TODO: A lot of time could be saved, by checking if all class counts are satisfied and breaking
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


def shuffle_h5(input_file, output_file, chunk_lines=32):
    input_h5 = h5py.File(input_file, 'r', libver='latest', swmr=True)
    n = len(input_h5['features'])
    print("Reading {} lines from dataset".format(n))

    samples = input_h5['features']  # [:1000]
    labels = input_h5['labels']  # [:1000]

    with h5py.File(output_file, 'w') as output_h5:
        logging.info("Shuffling {}. This will take approximately {}".format(input_file,
                                                                            datetime.timedelta(
                                                                                seconds=((0.0001175470 * n)))))
        joined = list(zip(samples, labels))
        random.shuffle(joined)
        samples, labels = zip(*joined)

        labels_n = len(labels[0])
        features = len(samples[0])

        dt = h5py.special_dtype(vlen=str)  # PY3

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
        logging.info("Writing shuffled file...")
        for i, (sample, label) in tqdm(enumerate(zip(samples, labels)), total=n):
            dset1[i] = sample
            dset2[i] = label

    logger.info("Saved raw shuffled file: {} with {} samples ({})".format(output_file, n,
                                                                          size(
                                                                              os.path.getsize(output_file))))


def preprocess_mongo(collection, raw_file, limit=None):
    logging.info("Retrieving from mongoDB")
    total = limit if limit else collection.estimated_document_count()

    logger.info("Gathering {} documents from mongoDB...".format(total))

    dt = h5py.special_dtype(vlen=str)  # PY3

    cursor = collection.find().limit(total)
    # cursor = collection.aggregate([{'$sample': {'size': total}}], allowDiskUse=True)
    # print("Shuffled documents")

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
                                   shape=(total, len(index_to_emotion)),
                                   compression="gzip",
                                   compression_opts=9,
                                   chunks=(fetch_and_read_batch_size, len(index_to_emotion)),
                                   dtype='int32')

        for i, document in tqdm(enumerate(cursor), total=total):
            raw_text = document['text']

            dset1[i] = raw_text
            label = make_multi_label_target(raw_text)
            if len(np.nonzero(label.numpy())[0]) == 0:
                bad_samples_count += 1
            dset2[i, :] = make_multi_label_target(raw_text)

    logging.info("Bad samples count: {}/{}".format(bad_samples_count, total))
    logger.info("Saved raw file: {} with {} samples ({})".format(raw_file, total, size(os.path.getsize(raw_file))))


def preprocess_mongo_balanced_class(collection, output_file, class_limit=None):
    logging.info("Retrieving samples with balance classing from mongoDB")
    min = None
    fetch_and_write_batch = 1024
    dt = h5py.special_dtype(vlen=str)  # PY3

    if not class_limit:
        min_class = None
        print("Counting minimum class")  # This is terribly slow
        for key, value in tqdm(index_to_emotion.items(), postfix="Class count"):
            count = collection.count_documents({"text": {'$regex': ".*[{}].*".format(value)}})
            if not min_class or count < min:
                min = count
                min_class = key

            print("Class {} Count {}".format(key, count))

        print("Counted documents. Minimum class {} has {} tweets".format(min_class, min))
    else:
        logging.info("Class limit is {}".format(class_limit))
        min = class_limit

    total = min * len(index_to_emotion)

    cursors = {}

    for key, value in index_to_emotion.items():
        # TODO: Use aggregate for shuffling
        cursors[key] = collection.find({"text": {'$regex': ".*[{}].*".format(value)}}).limit(min)

    is_multi_label = True

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
                                   shape=(total, len(index_to_emotion)),
                                   compression="gzip",
                                   compression_opts=9,
                                   chunks=(fetch_and_write_batch, len(index_to_emotion)),
                                   dtype='int32')
        i = 0
        for emotion_index, key in tqdm(enumerate(cursors), postfix="Class Iterate", total=total):
            cursor_name = key
            cursor = cursors[key]
            cursor.batch_size(fetch_and_write_batch)

            print("Gathering {}".format(cursor_name))
            for document in tqdm(cursor, postfix=cursor_name, total=min):
                raw_text = document['text']
                if is_multi_label:
                    # print("Target: {}".format(make_multi_label_target(raw_text)))
                    y = make_multi_label_target(raw_text)
                else:
                    y = make_target_with_unique_class(emotion_index)

                dset1[i] = raw_text
                dset2[i, :] = y

                i += 1

    logger.info(
        "Saved raw file: {} with {} samples ({})".format(output_file, total, size(os.path.getsize(output_file))))


def parse_config():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    return config, args


def preprocess(config):
    # This will scrap raw data from mongo server
    fromMongo = True
    processed_file_bow = config['models']['SVM']['dataset']  # '''''processed/tweets_bow.h5'
    processed_file_sequence = config['models']['LSTM']['dataset']  # 'processed/tweets_seq.h5'
    vocab_file = config["vocab_file"]  # ''processed/vocab.h5'
    labels_file = config["labels_file"]  # 'processed/labels.h5'

    raw_file = config["raw_file"]  # ''raw/tweets.h5'
    raw_shuffled_file = config["raw_shuffled"]  # 'raw/tweets_shuffled.h5'

    client = MongoClient(config["mongo"]["ip"], config["mongo"]["port"])
    # client = MongoClient("79.181.222.105", 27017)
    collection = client[config["mongo"]["db"]]["aug7"]  # .aug7

    vocab_size = config["vocab_size"]

    sequence_limit = config["sequence_limit"]
    batch_size = config["data_loader"]["batch_size"]

    if fromMongo:
        # preprocess_mongo_balanced_class(collection, raw_file, class_limit=None)  # Hungry face limit

        #preprocess_mongo(collection, raw_shuffled_file, limit=None)
        shuffle_h5(raw_file, raw_shuffled_file)


    # Should not use class_limit together with total_limit
    # TODO: Change to raw_shuffled when needed
    # preprocess_hdf5(raw_file,
    #                 processed_file_bow,
    #                 processed_file_sequence,
    #                 vocab_file,
    #                 labels_file,
    #                 vocab_size,
    #                 sequence_limit=sequence_limit,
    #                 chunk_lines=batch_size,
    #                 class_limit=1050000,
    #                 vocab_sample_limit = 1000000,
    #                 total_limit=None)

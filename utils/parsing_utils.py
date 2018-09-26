# from utils import contraction, correction, stop_words, stemmer, split_hashtag
import json
import os
import pandas
import re
from collections import Counter
import emoji
import numpy as np
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords, words
from nltk.stem.snowball import SnowballStemmer
from utils.index_to_emoji import idx2emoji

import cProfile, pstats, io
pr = cProfile.Profile()

script_dir = os.path.dirname(os.path.realpath(__file__))

# # stemming_____________________________________________________________________________________________________________
# stemmer = SnowballStemmer('english')
#
# # contraction__________________________________________________________________________________________________________
# # contraction = json.load(open('datasets/constants/contraction.json'))
# contraction = json.load(open(os.path.join(script_dir, 'constants/contraction.json')))
# # Adding capital case
# for key, val in list(contraction.items()):
#     contraction[key.upper()] = val.upper()
#
# # stopword_____________________________________________________________________________________________________________
# stop_words = stopwords.words('english')
# stop_words.remove('not')
#
#
# # correction__________________________________________________________________________________________________________
# def all_words(text):
#     return re.findall('\\w+', text.lower())
#
#
# WORDS = Counter(all_words(open(os.path.join(script_dir, 'constants/big.txt')).read()))
#
# def P(word, N=sum(WORDS.values())):
#     """Probability of `word`."""
#     return WORDS[word] / N
#
#
# def correction(word):
#     """Most probable spelling correct for word."""
#     return vocab.correction(word)
#     # return max(candidates(word), key=P)
#
#
# def candidates(word):
#     """Generate possible spelling corrections for word."""
#     return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
#
#
# def known(words):
#     """The subset of `words` that appear in the dictionary of WORDS."""
#     return set((w for w in words if w in WORDS))
#
#
# def edits1(word):
#     """All edits that are one edit away from `word`."""
#     letters = 'abcdefghijklmnopqrstuvwxyz'
#     splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#     deletes = [L + R[1:] for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
#     replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
#     inserts = [L + c + R for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)
#
#
# def edits2(word):
#     """All edits that are two edits away from `word`."""
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))
#
#
# # word dictionary______________________________________________________________________________________________________
# word_dictionary = list(set(words.words()))
# for alphabet in 'bcdefghjklmnopqrstuvwxyz':
#     word_dictionary.remove(alphabet)
#
# useless_two_letter_words = pandas.read_csv(os.path.join(script_dir, 'constants/useless_two_letter_words.csv'))
# for word in useless_two_letter_words:
#     word_dictionary.remove(word)
#
# useful_words = pandas.read_csv(os.path.join(script_dir, 'constants/useful_words.csv'))
# for word in useful_words:
#     word_dictionary.append(word)
#
# for key in contraction:
#     word_dictionary.append(key)
#
#
# # split hashtags_______________________________________________________________________________________________________
# def split_hashtag(hashtag):
#     found = False
#     for i in reversed(range(1, len(hashtag) + 1)):
#         if hashtag[:i] in vocab.WORDS:
#             found = True
#             if i == len(hashtag):
#                 if hashtag[:i] in contraction:
#                     return contraction[hashtag[:i]]
#                 else:
#                     return hashtag[:i]
#             else:
#                 child = split_hashtag(hashtag[i:])
#                 if child:
#                     return hashtag[:i] + ' ' + child
#     if not found:
#         return False


def make_bow_vector(sentence, word_to_ix):
    vec = np.zeros(len(word_to_ix))
    for word in sentence.split():
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec
    # return vec.view(1, -1)


def make_seq_vector(sentence, word_to_ix, sequence_limit):
    sequence = np.zeros(sequence_limit, dtype=np.int64)
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


def count_occurrences(text, reg):
    comp = re.compile(reg)
    return len(re.findall(comp, text))


# Creates a multi label, where each class is given 1 if it occurs in text, 0 otherwise
def make_multi_label_target(raw_text):
    vec = torch.LongTensor([(1 if count_occurrences(raw_text, reg) > 0 else 0) for reg in idx2emoji.values()])
    # vec = F.normalize(vec, p=2, dim=0)
    return vec


# Creates a multi label, where each class is given a proportion of its occurrences in the text
def make_multi_label_proportional_target(raw_text):
    vec = torch.FloatTensor([count_occurrences(raw_text, reg) for reg in idx2emoji.values()])
    vec = F.normalize(vec, p=2, dim=0)
    return vec


# Creates a label for a given class (index of index_to_emotion)
def make_target_with_unique_class(index):
    vec = np.zeros(len(idx2emoji))
    vec[index] = 1

    return vec


# Creates a label for the maximal class
def make_unique_target_for_maximal_class(raw_text):
    count_list = [count_occurrences(raw_text, reg) for reg in idx2emoji.values()]
    return None if len(np.nonzero(count_list)[0]) < 1 else np.argmax(count_list)


# Returns [labels] for all classes that are present in text
def make_targets_for_classes(raw_text):
    count_list = [count_occurrences(raw_text, reg) for reg in idx2emoji.values()]
    return np.flatnonzero(count_list)

# Creates a label for the maximal class
def make_target(raw_text):
    count_array = np.array([count_occurrences(raw_text, reg) for reg in idx2emoji.values()])
    vec = np.zeros(len(idx2emoji))
    vec[np.argmax(count_array)] = 1
    return vec

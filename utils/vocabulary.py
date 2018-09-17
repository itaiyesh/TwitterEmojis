import gensim
import re
from collections import Counter
from .deep_moji_parser import *
import itertools
from bs4 import BeautifulSoup
import numpy as np

class Vocabulary():
    def __init__(self , debug = False):
        if not debug:
            file1 = 'C:\\Users\iyeshuru\Downloads\word2vec_twitter_model.bin'
            file1 = 'C:\\Users\iyeshuru\Downloads\GoogleNews-vectors-negative300.bin.gz'

            model = gensim.models.KeyedVectors.load_word2vec_format(file1,binary = True, unicode_errors='ignore')
            words = model.index2word

            w_rank = {}
            for i,word in enumerate(words):
                w_rank[word] = i

            self.WORDS = w_rank

        self.debug = debug

    def words(self, text): return re.findall(r'\w+', text.lower())

    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.WORDS.get(word, 0)

    def correction(self, word):
        if self.debug:
            return word
        "Most probable spelling correct for word."
        return max( self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return ( self.known([word]) or  self.known( self.edits1(word)) or  self.known( self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in  self.edits1(e1))

    # % = ".."
    def is_punctuation(self, word):
        return set(word).issubset("?!%")

    # !!? -> !?
    # ?! -> !?
    def normalize_punctuation(self, word):
        return ''.join(list(set(word)))

    def clean(self, tweet_text):
        # tweet_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet_text))
        # tweet_text = ' '.join(tokenize(tweet_text))
        # # Can't store "." keys in hdf5 so we have to convert it to \.
        # tweet_text = re.sub(r'\.', '\.', tweet_text)
        # return tweet_text


        # ignore non ascii characters
        # TODO: Lower + upper

        # TODO: Remove pure digits
        tweet_text = tweet_text.encode('ascii', 'ignore')
        # tweet_text = tweet_text.lower()

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

        # .. -> %
        # This will fuck up word to vec
        # tweet_text = re.sub(r'\.\.', '%', tweet_text)



        # Space out words from punctuation marks (excluding '.' and ',')
        # hello? -> hello ?
        tweet_text = re.sub('([!?]+)', r' \1', tweet_text)

        # TODO: This part takes forever!!
        # useless_hashtag = ['tcot', 'tlot', 'ucot', 'p2b', 'p2', 'ccot', 'pjnet', 'gop', 'nra']
        # split_tweet_text = ''
        # for word in tweet_text.split():
        #
        #     if is_punctuation(word):
        #         split_tweet_text += ' ' + normalize_punctuation(word)
        #
        #     # split hashtags
        #     elif word.startswith('#'):
        #         continue
        #         word = smart_cap(word[1:])
        #
        #         split_words = split_hashtag(word) if word not in useless_hashtag else ' '
        #         if split_words:
        #             split_tweet_text += (' ' + correction(word) + ' ' + split_words)
        #         else:
        #             split_tweet_text += (' ' + correction(word))
        #     else:
        #
        #         word = smart_cap(word)
        #
        #         #expand contractions
        #         if word in contraction:
        #             split_tweet_text += ' ' + contraction[word]
        #         else:
        #             split_tweet_text += ' ' + correction(word)
        #
        # tweet_text = split_tweet_text

        # print(tweet_text)
        # remove special char (except #) and contraction for hash tag
        # tweet_text = re.sub('[^0-9a-zA-Z!?\.]', ' ', tweet_text)


        # Can't store "." keys in hdf5 so we have to convert it to \.
        tweet_text = re.sub(r'\.', '\.', tweet_text)


        # TODO: These stuff are not so good for LSTM
        # tokenize
        # TODO: Whats this?
        # words = word_tokenize(tweet_text)
        # words = tweet_text.split()

        # remove stopwords
        # words = [stemmer.stem(w) for w in words if not w in stop_words]
        # words = [w for w in words if not w in stop_words]

        # join the words
        # tweet_text = " ".join(words)

        return tweet_text

    def is_mostly_capital(self, word):
        half = len(word)/2.0
        return float(sum(1 for c in word if c.isupper())) + np.finfo(float).eps > half

    def smart_cap(self, word):
        # TODO: Make sure He remains He and not HE (strictly greater than 0.5)
        return word.upper() if self.is_mostly_capital(word) else word.lower()

#TODO: Put inside class, this takes lots of memory (for multi process esp)

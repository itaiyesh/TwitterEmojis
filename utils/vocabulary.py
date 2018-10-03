import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import re
from collections import Counter
from .deep_moji_parser import *
import itertools
from bs4 import BeautifulSoup
import numpy as np
import os, json
from abc import ABCMeta, abstractmethod

from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import FreqDist
from utils import *
import random

class Vocabulary():

    def __init__(self, config = None,
                 samples = None,
                 output_vocab_file = None,
                 tfidf_limit = None,
                 w2v_limit = None,
                 vocab_sample_limit = None, sequence_limit_percentile = None, logger = None, debug = True, vocab_file = None, w2v_file = None):


        if not debug and w2v_file:
            model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file,binary = True, unicode_errors='ignore')
            words = model.index2word

            w_rank = {}
            for i,word in enumerate(words):
                w_rank[word] = i

            self.words = w_rank

        # Contraction file
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.contraction = json.load(open(os.path.join(script_dir, 'constants/contraction.json')))

        # Adding capital case
        for key, val in list(self.contraction.items()):
            self.contraction[key.upper()] = val.upper()

        self.debug = debug


        if vocab_file:
            self.load_vocab_file(vocab_file)
        else:
            self.prepare_vocab(config ,
                 samples,
                 output_vocab_file,
                 tfidf_limit,
                 w2v_limit ,
                 vocab_sample_limit , sequence_limit_percentile , logger )

    def load_vocab_file(self, vocab_file):
        word_to_ix = {}
        with h5py.File(vocab_file, 'r') as h5f:
            for ds in h5f.keys():
                word_to_ix[ds] = int(h5f[ds].value)

        self.word_to_ix = word_to_ix

    def prepare_vocab(self,config ,
                 samples,
                 output_vocab_file,
                 tfidf_limit,
                 w2v_limit ,
                 vocab_sample_limit , sequence_limit_percentile , logger ):

        logger.info("Building vocabulary")

        all_sentences = []
        if vocab_sample_limit:
            logger.info("Limiting vocab build up to {} samples".format(vocab_sample_limit))
            # vocab_samples = samples[:vocab_sample_limit]


            # First approach - doesnt shuffle indices. Partitions sampling to continuous chunks.
            # vocab_samples = np.concatenate((samples[:int(vocab_sample_limit/2)],samples[-int(vocab_sample_limit/2):]), axis=0)

            # Second approach - shuffle. years...
            # This approach first shuffles indices then retrieves tweets
            # logging.info("Generating random indices for tweets.")
            # indices =random.sample(list(range(0, len(samples))),vocab_sample_limit)
            # logging.info("Sorting indices...")
            # indices = sorted(indices)
            # logging.info("Retrieving {} random tweets...".format(vocab_sample_limit))
            # vocab_samples = samples[indices]

            chunk_size = 1024# take from preconfig 'fetch_and_write_size...'
            logging.info("Reading random chunks of size {}".format(chunk_size))
            chunks = int(np.floor(vocab_sample_limit / chunk_size))
            indices = random.sample(list(range(0,int(np.floor(len(samples)/chunk_size)))), chunks)
            # vocab_samples = samples[:1] #
            for i in tqdm(indices, total=len(indices)):
                #TODO: Allocate ahead of time
                for sample in samples[i*chunk_size:(i+1)*chunk_size]:
                    all_sentences.append(self.clean(sample[0]))
                # vocab_samples = np.concatenate([vocab_samples, samples[i*chunk_size:(i+1)*chunk_size]])

        else:
            for sample in samples:
                all_sentences.append(self.clean(sample[0]))  # for some reason, we need this indexing
            # vocab_samples = samples

        # logging.info("Cleaning sentences etc...")
        # #TODO: tmp hdf5 file to hold words. memory limit etc...
        # for sample in tqdm(vocab_samples):
        #     all_sentences.append(self.clean(sample[0]))  # for some reason, we need this indexing

        logger.info("Calculating {:.1f}% percentile of lengths".format(sequence_limit_percentile))
        sentence_tokens = [sentence.split() for sentence in all_sentences]
        sentence_tokens = [(tokens, len(tokens)) for tokens in sentence_tokens]
        _, lengths = zip(*sentence_tokens)
        sequence_limit = np.percentile(lengths, sequence_limit_percentile)
        logger.info("{:.1f}% percentile is {}".format(sequence_limit_percentile, sequence_limit))
        # num_sentences = len(sentence_tokens)
        # filter(lambda sentence_tokens: sentence_tokens[1] <= sequence_limit, sentence_tokens)
        # logging.info("Removed {} sentence. Remaining: {}".format(num_sentences-len(sentence_tokens), len(sentence_tokens)))
        # sentence_tokens,_ = zip(*sentence_tokens)

        # Method 0
        sentences = [sentence.split() for sentence in all_sentences]
        model = gensim.models.Word2Vec(sentences, min_count=5, workers=config['num_workers'], sg=1) #sg=1 for skipgram

        words = []
        norms = []
        freqs = []
        norm_per_freqs = []
        for word in model.wv.vocab:
            words.append(word)
            norms.append(np.linalg.norm(model.wv.get_vector(word)))
            freqs.append(model.wv.vocab[word].count)
            norm_per_freqs.append(norms[-1] / freqs[-1])

        df = pd.DataFrame({"words": words, "norms": norms, "freqs": freqs, "npf": norm_per_freqs})
        df = df.sort_values(by=['norms', 'freqs'], ascending=False).head(w2v_limit)
        w2v_words = df.words.get_values()
        logger.info("Running TF-IDF on collected words...")

        # Method 1
        # TF-IDF scoring of weight, adapted from
        # http://www.ultravioletanalytics.com/blog/tf-idf-basics-with-pandas-scikit-
        # More info on sklearn.feature_extraction.text.TfidfTransformer
        tvec = TfidfVectorizer(min_df=0.0, max_df=.9, stop_words='english', ngram_range=(1, 1))
        tvec_weights = tvec.fit_transform(all_sentences)
        weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()
        weights_df = pd.DataFrame({'term': tvec.get_feature_names(), 'weight': weights})
        tfidf_words = weights_df.sort_values(by='weight', ascending=False).head(tfidf_limit)
        tfidf_words = list(tfidf_words['term'])

        # Method 2
        # all_sentences_tokens = [sentence.split() for sentence in all_sentences]
        # dictionary = Dictionary.from_documents(all_sentences_tokens)
        # corpus = [dictionary.doc2bow(doc) for doc in all_sentences_tokens]
        # tfidf = TfidfModel(corpus, id2word=dictionary)
        # word_values = []
        # for bow in corpus:
        #     word_values += [w for w in tfidf[bow]]
        #
        # word_values.sort(key=lambda x: x[1], reverse=True)
        # tfidf_words = [tfidf.id2word[w[0]] for w in word_values]
        # tfidf_words = list(set(tfidf_words))[:vocab_size]

        # Adding most frequent k as suggested by
        # https://openreview.net/pdf?id=Bk8N0RLxx
        logger.info("Calculating most frequent words...")
        all_words = []
        for sentence in all_sentences:
            for word in sentence.split():
                all_words.append(word)
        # This is using simple frequency
        freq = FreqDist(all_words)
        most_frequent_words = [w[0] for w in freq.most_common(config['frequent_words_limit'])]
        # print(most_frequent_words)

        word_to_ix = {}

        # Note: It's important to keep pad indexed at 0
        best_words = ['<pad>'] + list(set(tfidf_words).union(set(w2v_words)).union(set(most_frequent_words))) + [
            '<unk>']

        for word in best_words:
            word_to_ix[word] = len(word_to_ix)

        try:
            logger.info("Removing {}".format(output_vocab_file))
            os.remove(output_vocab_file)
        except:
            logger.info("Could not remove file")

        logger.info("Writing vocab file")

        with h5py.File(output_vocab_file, 'w') as h5_bow:
            for k, v in tqdm(word_to_ix.items()):
                # if k == 'b':
                #     print(k)
                # print(".{}.".format(k))
                h5_bow.create_dataset(k, data=v)

        logger.info("Saved vocab file: {} with {} words ({})".format(output_vocab_file, len(word_to_ix),
                                                                     size(os.path.getsize(output_vocab_file))))
        self.word_to_ix = word_to_ix

    # def __init__(self , debug = False, w2v_file = None):
    #     if not debug and w2v_file:
    #         model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file,binary = True, unicode_errors='ignore')
    #         words = model.index2word
    #
    #         w_rank = {}
    #         for i,word in enumerate(words):
    #             w_rank[word] = i
    #
    #         self.words = w_rank
    #
    #     # Contraction file
    #     script_dir = os.path.dirname(os.path.realpath(__file__))
    #     self.contraction = json.load(open(os.path.join(script_dir, 'constants/contraction.json')))
    #
    #     # Adding capital case
    #     for key, val in list(self.contraction.items()):
    #         self.contraction[key.upper()] = val.upper()
    #
    #     self.debug = debug

    def words(self, text): return re.findall(r'\w+', text.lower())


    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.words.get(word, 0)

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
        return set(w for w in words if w in self.words)

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
        tweet_text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet_text))
        # remove html tags
        tweet_text = BeautifulSoup(tweet_text, 'html.parser').get_text()
        # remove 'RT' text
        tweet_text = re.sub(r'(^| )rt ', ' ', tweet_text)
        # remove links
        tweet_text = re.sub(r'https?:\/\/\S*', ' ', tweet_text)

        # tweet_text = ' '.join(tokenize(tweet_text))
        words = []
        for word in tokenize(tweet_text):
            #TODO: Test test test
            if word.startswith("#") or word.startswith("@"):
                continue

            word = self.smart_cap(word)
            if word in self.contraction:
                words += self.contraction[word].split()
            elif self.is_punctuation(word):
                words.append(self.normalize_punctuation(word))
            else:
                words.append(word)

        tweet_text = ' '.join(words)

        # Note, we remove emojis for training
        tweet_text = re.sub('[^0-9a-zA-Z!?\.]', ' ', tweet_text)

        # Can't store "." keys in hdf5 so we have to convert it to \.
        tweet_text = re.sub(r'\.', '\.', tweet_text)

        return tweet_text

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
        # This will fuck up word to vec(?)
        # tweet_text = re.sub(r'\.\.', '%', tweet_text)

        # Space out words from punctuation marks (excluding '.' and ',')
        # hello? -> hello ?
        tweet_text = re.sub('([!,?]+)', r' \1', tweet_text)

        # Smart cap

        words = []
        for word in tweet_text.split():
            word = self.smart_cap(word)
            if word in self.contraction:
                words += self.contraction[word].split()
            elif self.is_punctuation(word):
                words.append(self.normalize_punctuation(word))
            else:
                words.append(word)

        tweet_text = ' '.join(words)

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
        # TODO: This fucks up w2v
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

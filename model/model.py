from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import numpy as np
import h5py
import logging
from tqdm import tqdm
from numpy import pi
from sklearn.naive_bayes import MultinomialNB
import re

class LSTM(BaseModel):

    def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTM, self).__init__(general_config)

        self.batch_size = general_config['data_loader']['batch_size']

        self.hidden_dim = model_config['hidden_dim']
        self.embed_dim =  model_config['embedding_dim']

        #TODO: remove this
        self.use_gpu = True

        self.word_embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # print("{} {} {}".format(1, self.batch_size, self.hidden_dim))
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def on_batch(self):
        self.hidden = self.init_hidden()

    def forward(self, sentence):
        # print("Shape sentence: {}".format(sentence.shape))
        # print(sentence.shape)
        sentence = sentence.t()
        # print(sentence.shape)
        embeds = self.word_embeddings(sentence)
        # print(embeds.shape)
        x = embeds.view(len(sentence), self.batch_size, -1)
        # print(x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        y = self.hidden2label(lstm_out[-1])#.float()
        # print(y)
        # print(y.shape)
        return F.log_softmax(y)

# class LSTM2(BaseModel):
#
#     def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
#         super(LSTM2, self).__init__(general_config)
#
#         self.batch_size = general_config['data_loader']['batch_size']
#
#         self.hidden_dim = model_config['hidden_dim']
#         self.embed_dim =  model_config['embedding_dim']
#
#         #TODO: remove this
#         self.use_gpu = True
#
#         self.word_embeddings = nn.Embedding(vocab_size, self.embed_dim)
#         self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True)
#         self.hidden2label = nn.Linear(self.hidden_dim, label_size)
#         self.hidden = self.init_hidden()
#
#     def init_hidden(self):
#         # print("{} {} {}".format(1, self.batch_size, self.hidden_dim))
#         if self.use_gpu:
#             h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
#             c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
#         else:
#             h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
#             c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
#         return (h0, c0)
#
#     def on_batch(self):
#         self.hidden = self.init_hidden()
#
#     def forward(self, data):
#         sentence, lengths = data
#
#         # print(sentence.shape)
#         # print(len(lengths))
#
#         embedding = self.word_embeddings(sentence)
#         # print("Embedding: {}".format(embedding))
#         # print("Embedding size: {}".format(embedding.shape))
#         # print("Lengths: {}".format(lengths))
#         packed_embedding = pack_padded_sequence(embedding, lengths, batch_first=True)
#         # print("Packed embedding: {}".format(packed_embedding))
#         # out, _ = self.gru(packed_embedding)
#         lstm_out, self.hidden = self.lstm(packed_embedding, self.hidden)
#         # print("lstm_out: {}".format(lstm_out))
#         # out, lengths = pad_packed_sequence(lstm_out, batch_first=True)
#         # print("out: {}".format(out.shape))
#         # print("out: {}".format(out))
#         # print("Hidden: {}".format(self.hidden[0]))
#         # Since we are doing classification, we only need the last
#
#         # output from RNN
#
#         # lengths = [l - 1 for l in lengths]
#         #
#         # last_output = out[lengths, range(len(lengths))]
#
#         logits = self.hidden2label(self.hidden[0]).squeeze()
#         # print(logits)
#         # print(logits.shape)
#         return F.log_softmax(logits)

def generate_embeddings(vocab_size, w2v_file, vocab_file):
    logging.info("Loading w2v file: {}".format(w2v_file))
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file,
                                                          binary=True,
                                                          unicode_errors='ignore')
    embed_dim = w2v.vector_size


    word_embeddings = nn.Embedding(vocab_size, embed_dim)

    logging.info("Updating vocab vectors...")
    random_vectors_count = 0
    for word in tqdm(vocab_file.keys()):
        index = int(vocab_file[word].value)

        # Can't store "." keys in hdf5 so we had to convert it to \.
        # \. -> .
        word = re.sub(r'\\\.', '.', word)

        try:
            word_embeddings.weight.data[index, :].set_(torch.FloatTensor(w2v.wv[word]))

        except KeyError:
            random_vectors_count+=1
            word_embeddings.weight.data[index, :].set_(
                torch.FloatTensor(np.random.normal(scale=0.6, size=(embed_dim,))))

    # TODO: Uncomment
    word_embeddings.weight.requires_grad = False

    if random_vectors_count > 0:
        logging.warning("{:.1f}% words do not exist in the pretrained embeddings".format(
            100*(float(random_vectors_count) / len(vocab_file))))

    return word_embeddings, embed_dim


class LSTM2(BaseModel):



    def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTM2, self).__init__(general_config)

        self.batch_size = general_config['data_loader']['batch_size']

        self.hidden_dim = model_config['hidden_dim']

        #TODO: remove this
        self.use_gpu = True

        if 'word2vec' in general_config:
            w2v_file = general_config['word2vec_files'][general_config['word2vec']]
            vocab_file = h5py.File(general_config['vocab_file'], 'r', libver='latest', swmr=True)
            self.word_embeddings, embed_dim = generate_embeddings(vocab_size,w2v_file, vocab_file)
        else:
            embed_dim = model_config['embedding_dim']
            self.word_embeddings =  nn.Embedding(vocab_size,embed_dim)


        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(self.hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def on_batch(self):
        self.hidden = self.init_hidden()

    def forward(self, data):
        sentence, lengths = data

        embedding = self.word_embeddings(sentence)

        packed_embedding = pack_padded_sequence(embedding, lengths, batch_first=True)

        lstm_out, self.hidden = self.lstm(packed_embedding, self.hidden)

        logits = self.hidden2label(self.hidden[0]).squeeze()

        return F.log_softmax(logits)


#with dropout
class biLSTMDO(BaseModel):
    def __init__(self, vocab_size, label_size, run_config, model_config,pre_config):  # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(biLSTMDO, self).__init__(run_config)

        self.batch_size = run_config['data_loader']['batch_size']
        self.label_size = label_size
        self.hidden_dim = model_config['hidden_dim']

        # TODO: remove this
        self.use_gpu = True

        if 'word2vec' in run_config:
            w2v_file = pre_config['word2vec_files'][run_config['word2vec']]
            vocab_file = h5py.File(pre_config['vocab_file'], 'r', libver='latest', swmr=True)
            self.word_embeddings, embed_dim = generate_embeddings(vocab_size,w2v_file, vocab_file)
        else:
            embed_dim = model_config['embedding_dim']
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

        self.embed_dropout = nn.Dropout2d(0.4)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.final_dropout = nn.Dropout2d(0.3)
        self.hidden2label = nn.Linear(self.hidden_dim, label_size)
        self.hidden = self.init_hidden()

        # To be used in pretraining
        # self.l2 = None
        # self.is_pretrained = False

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def on_batch(self):
        self.hidden = self.init_hidden()

    # def is_pretrainable(self):
    #     return True

    # def finetune(self,  output_labels, model = None):
    #     # Option - load weights from another model
    #     if model:
    #         self.word_embeddings.weight.data = model.word_embeddings.weight.data
    #         self.hidden2label.weight.data = model.hidden2label.weight.data
    #         self.hidden = model.hidden
    #
    #     self.is_pretrained = True
    #     self.l2 = nn.Linear(self.label_size, output_labels)

    def forward(self, data):
        sentence, lengths = data

        embedding = self.word_embeddings(sentence)

        embedding = self.embed_dropout(embedding)

        packed_embedding = pack_padded_sequence(embedding, lengths, batch_first=True)

        lstm_out, _ = self.lstm(packed_embedding, self.hidden)
        _, self.hidden = self.lstm2(lstm_out, self.hidden)

        x = self.final_dropout(self.hidden[0][0] + self.hidden[0][1])

        x = self.hidden2label(x).squeeze()
        # if self.is_pretrained:
        #     x = self.l2(x)
        return F.log_softmax(x) #dim = 0?


class biLSTMDOPretrained(biLSTMDO):
    def __init__(self, vocab_size, label_size, run_config, model_config, pre_config,finetune_labels_size):
        super(biLSTMDOPretrained, self).__init__(vocab_size, label_size, run_config, model_config,pre_config)
        self.l2 = nn.Linear(label_size, finetune_labels_size)

    def load_pretrained(self, model):
        #TODO: Note, the embeddings arent transfering right for some reason...shape is 27
        self.word_embeddings.weight.data = model.word_embeddings.weight.data
        self.hidden2label.weight.data = model.hidden2label.weight.data
        self.hidden = model.hidden

        #TODO: Test with and without
        self.word_embeddings.weight.requires_grad = False
        self.hidden2label.weight.requires_grad = False


    def forward(self, data):
        y = super().forward(data)
        y = self.l2(y)
        return F.log_softmax(y)

class biLSTM(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(biLSTM, self).__init__(general_config)

        self.batch_size = general_config['data_loader']['batch_size']

        self.hidden_dim = model_config['hidden_dim']

        #TODO: remove this
        self.use_gpu = True

        if 'word2vec' in general_config:
            w2v_file = general_config['word2vec_files'][general_config['word2vec']]
            vocab_file = h5py.File(general_config['vocab_file'], 'r', libver='latest', swmr=True)
            self.word_embeddings, embed_dim = generate_embeddings(vocab_size,w2v_file, vocab_file)
        else:
            embed_dim = model_config['embedding_dim']
            self.word_embeddings =  nn.Embedding(vocab_size,embed_dim)

        # self.word_embeddings, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix, True)


        # self.word_embeddings = nn.Embedding.from_pretrained(weights)

        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2* self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True)

        self.hidden2label = nn.Linear(self.hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def on_batch(self):
        self.hidden = self.init_hidden()

    def forward(self, data):
        sentence, lengths = data

        embedding = self.word_embeddings(sentence)

        packed_embedding = pack_padded_sequence(embedding, lengths, batch_first=True)

        lstm_out, _= self.lstm(packed_embedding, self.hidden)
        _, self.hidden = self.lstm2(lstm_out, self.hidden)

        logits = self.hidden2label(self.hidden[0][0]+self.hidden[0][1]).squeeze()

        return F.log_softmax(logits)

class LSTM2Layer(BaseModel):

    def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTM2Layer, self).__init__(general_config)

        self.batch_size = general_config['data_loader']['batch_size']

        self.hidden_dim = model_config['hidden_dim']

        #TODO: remove this
        self.use_gpu = True

        if 'word2vec' in general_config:
            w2v_file = general_config['word2vec_files'][general_config['word2vec']]
            vocab_file = h5py.File(general_config['vocab_file'], 'r', libver='latest', swmr=True)
            self.word_embeddings, embed_dim = generate_embeddings(vocab_size,w2v_file, vocab_file)
        else:
            embed_dim = model_config['embedding_dim']
            self.word_embeddings =  nn.Embedding(vocab_size,embed_dim)


        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.hidden2label = nn.Linear(self.hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def on_batch(self):
        self.hidden = self.init_hidden()

    def forward(self, data):
        sentence, lengths = data

        embedding = self.word_embeddings(sentence)

        packed_embedding = pack_padded_sequence(embedding, lengths, batch_first=True)

        lstm_out, _= self.lstm(packed_embedding, self.hidden)
        _, self.hidden = self.lstm2(lstm_out, self.hidden)

        logits = self.hidden2label(self.hidden[0]).squeeze()

        return F.log_softmax(logits)


class SVM(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, model_config,pre_config):
        super(SVM, self).__init__(general_config)
        self.config = general_config
        self.l1 = nn.Linear(vocab_size, label_size)

    def forward(self, x):
        #TODO: Pass thru embedding here!
        y = self.l1(x.float())
        return F.log_softmax(y)


# svm with embedding
# class SVME(BaseModel):
#     def __init__(self, vocab_size, label_size, run_config, model_config, pre_config):
#         super(SVME, self).__init__(run_config)
#         self.config = run_config
#         self.word_embeddings = nn.Embedding(vocab_size,300)
#         self.l1 = nn.Linear(300, label_size)
#         self.sequence_limit = pre_config['sequence_limit']
#
#
#
#     def reverse(self, y):
#         # y = index of class
#         # print("predicting: {}".format(y))
#         y_hot = np.zeros(self.l1.weight.shape[0])
#         y_hot[y]=1
#         # print(y_hot.shape)
#         t1 = y_hot-self.l1.bias.cpu().detach().numpy()
#         # print(t1.shape)
#         L_inv = np.linalg.pinv(self.l1.weight.cpu().detach().numpy())
#         # print(L_inv.shape)
#         t2 = np.dot(L_inv,t1)
#
#         print(t2.shape)
#         E_inv = np.linalg.pinv(self.word_embeddings.weight.cpu().detach().numpy())
#         E_inv = np.transpose(E_inv)
#         # print(E_inv.shape)
#         t3 = np.dot(E_inv,t2)
#         # print(t3.shape)
#         return t2,t3
#
#
#     def forward(self, x):
#         #TODO: Pass thru embedding here!
#         # print(x.shape)
#         # must preserve order!
#         vec = x.data.cpu().numpy()
#         bach_size = vec.shape[0]
#
#         limit = self.sequence_limit  #words per sentence
#         batch = np.zeros((bach_size,limit))
#         for sentence_idx, word in enumerate(vec):
#             sentence = np.zeros(limit, dtype= word.dtype)
#             i = 0
#             while np.any(word) and i < limit:
#                 non_zero_index = np.nonzero(word)[0][0]
#                 word[non_zero_index]-=1
#                 sentence[i] = non_zero_index
#                 i+=1
#             batch[sentence_idx] = sentence
#
#         embeddings = self.word_embeddings(torch.LongTensor(batch).cuda())
#
#         embeddings = embeddings.mean(1)
#
#         y = self.l1(embeddings)
#
#
#         return F.log_softmax(y)
class SVMEw2v(BaseModel):
        def __init__(self, vocab_size, label_size, run_config, model_config, pre_config):
            super(SVMEw2v, self).__init__(run_config)
            self.config = run_config
            if 'word2vec' in run_config:
                w2v_file = pre_config['word2vec_files'][run_config['word2vec']]
                vocab_file = h5py.File(pre_config['vocab_file'], 'r', libver='latest', swmr=True)
                self.word_embeddings, embed_dim = generate_embeddings(vocab_size, w2v_file, vocab_file)
                #TODO: Do that in other places
                # self.word_embeddings.weight.requires_grad = True
            else:
                embed_dim = model_config['embedding_dim']
                self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

            self.l1 = nn.Linear(embed_dim, label_size)
            self.sequence_limit = pre_config['sequence_limit']
            logging.info(model_config)

        def reverse(self, y):
            # y = index of class
            # print("predicting: {}".format(y))
            y_hot = np.zeros(self.l1.weight.shape[0])
            y_hot[y] = 1
            # print(y_hot.shape)
            t1 = y_hot - self.l1.bias.cpu().detach().numpy()
            # print(t1.shape)
            L_inv = np.linalg.pinv(self.l1.weight.cpu().detach().numpy())
            # print(L_inv.shape)
            t2 = np.dot(L_inv, t1)

            # print(t2.shape)
            E_inv = np.linalg.pinv(self.word_embeddings.weight.cpu().detach().numpy())
            E_inv = np.transpose(E_inv)
            # print(E_inv.shape)
            t3 = np.dot(E_inv, t2)
            # print(t3.shape)
            return t2, t3

        def forward(self, x):
            # TODO: Pass thru embedding here!
            # print(x.shape)
            # must preserve order!
            vec = x.data.cpu().numpy()
            # emb_dim = self.word_embeddings.weight.shape[1]
            bach_size = vec.shape[0]

            # average_vec = np.zeros((bach_size, emb_dim))

            limit = self.sequence_limit  # words per sentence
            batch = np.zeros((bach_size, limit))
            for sentence_idx, word in enumerate(vec):
                sentence = np.zeros(limit, dtype=word.dtype)
                i = 0
                while np.any(word) and i < limit:
                    non_zero_index = np.nonzero(word)[0][0]
                    word[non_zero_index] -= 1
                    sentence[i] = non_zero_index
                    i += 1
                batch[sentence_idx] = sentence

            embeddings = self.word_embeddings(torch.LongTensor(batch).cuda())
            # print(embeddings.shape)
            embeddings = embeddings.mean(1)

            y = self.l1(embeddings)

            return F.log_softmax(y)

class SVME(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, model_config, pre_config):
        super(SVME, self).__init__(general_config)
        self.config = general_config
        self.batch_size = general_config['data_loader']['batch_size']
        self.word_embeddings = nn.Embedding(vocab_size,300)
        self.l1 = nn.Linear(300, label_size)
        self.sequence_limit = pre_config['sequence_limit']
        logging.info(model_config)

    def reverse(self, y):
        # y = index of class
        # print("predicting: {}".format(y))
        y_hot = np.zeros(self.l1.weight.shape[0])
        y_hot[y]=1
        # print(y_hot.shape)
        t1 = y_hot-self.l1.bias.cpu().detach().numpy()
        # print(t1.shape)
        L_inv = np.linalg.pinv(self.l1.weight.cpu().detach().numpy())
        # print(L_inv.shape)
        t2 = np.dot(L_inv,t1)

        # print(t2.shape)
        E_inv = np.linalg.pinv(self.word_embeddings.weight.cpu().detach().numpy())
        E_inv = np.transpose(E_inv)
        # print(E_inv.shape)
        t3 = np.dot(E_inv,t2)
        # print(t3.shape)
        return t2,t3


    def forward(self, x):
        #TODO: Pass thru embedding here!
        # print(x.shape)
        # must preserve order!
        vec = x.data.cpu().numpy()
        # emb_dim = self.word_embeddings.weight.shape[1]
        bach_size = vec.shape[0]

        # average_vec = np.zeros((bach_size, emb_dim))

        limit = self.sequence_limit  #words per sentence
        batch = np.zeros((bach_size,limit))
        for sentence_idx, word in enumerate(vec):
            sentence = np.zeros(limit, dtype= word.dtype)
            i = 0
            while np.any(word) and i < limit:
                non_zero_index = np.nonzero(word)[0][0]
                word[non_zero_index]-=1
                sentence[i] = non_zero_index
                i+=1
            batch[sentence_idx] = sentence

        embeddings = self.word_embeddings(torch.LongTensor(batch).cuda())
        # print(embeddings.shape)
        embeddings = embeddings.mean(1)

        y = self.l1(embeddings)

        return F.log_softmax(y)

# use for testing
class SVMEPretrained(SVME):# svm with embedding
    def __init__(self, vocab_size, label_size, run_config, model_config, pre_config, finetune_labels_size):
        super(SVMEPretrained, self).__init__(vocab_size, label_size, run_config, model_config, pre_config)
        self.config = run_config
        self.batch_size = run_config['data_loader']['batch_size']

        # Experiment with freezing different layers
        # self.word_embeddings.weight.requires_grad = True
        self.do1 = nn.Dropout(0.4)
        # self.l1.weight.requires_grad = True
        self.do2 = nn.Dropout(0.4)
        self.l2 = nn.Linear(label_size, finetune_labels_size)

    def load_pretrained(self, model):
        self.word_embeddings.weight.data = model.word_embeddings.weight.data
        self.l1.weight.data = model.l1.weight.data
        #TODO freze layers here!
        print("Layers for SVMEPretrained not frozen!")

    def forward(self, x):
        # Short version. no dropout
        # y = super().forward(x)
        #
        # y = self.l2(y)
        #
        # return F.log_softmax(y)

        vec = x.data.cpu().numpy()
        bach_size = vec.shape[0]
        limit = self.sequence_limit  #words per sentence
        batch = np.zeros((bach_size,limit))
        for sentence_idx, word in enumerate(vec):
            sentence = np.zeros(limit, dtype= word.dtype)
            i = 0
            while np.any(word) and i < limit:
                non_zero_index = np.nonzero(word)[0][0]
                word[non_zero_index]-=1
                sentence[i] = non_zero_index
                i+=1
            batch[sentence_idx] = sentence

        embeddings = self.word_embeddings(torch.LongTensor(batch).cuda())
        embeddings = self.do1(embeddings)
        embeddings = embeddings.mean(1)
        y = self.l1(embeddings)
        y = self.do2(y)
        y = self.l2(y)
        return F.log_softmax(y)

class CNN(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, model_config):
        super(CNN, self).__init__(general_config)

        self.nembedding = 300 # model_config['embedding_dim']
        kernel_num = 3
        kernel_sizes = [3,4, 5]
        label_size = label_size
        dropout = 0.3
        use_pretrain = False
        embed_matrix = None
        embed_freeze = False

        self.embedding = nn.Embedding(vocab_size, self.nembedding)
        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)
        self.in_channel = 1
        self.out_channel = kernel_num
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.convs1 = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (K, self.nembedding))
                                     for K in self.kernel_sizes])  # kernel_sizes,like (3,4,5)

        self.dropout = nn.Dropout(dropout)
        """
        in_feature=len(kernel_sizes)*kernel_num,because we concatenate 
        """
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, label_size)

    def forward(self, sequences):
        # padded_sentences, lengths = pad_packed_sequence(sequences, padding_value=int(0),
        #                                                 batch_first=True)  # set batch_first true
        #padded sequence
        x = self.embedding(sequences)  # batch_size*num_word*nembedding

        x = x.unsqueeze(1)  # (batch_size,1,num_word,nembedding)   1 is in_channel

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # a list containing (batch_size,out_channel,W)

        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in
             x]  # max_pool1d(input, kernel_size),now x is a list of (batch_size,out_channel)

        x = torch.cat(x, dim=1)  # concatenate , x is batch_size,len(kernel_sizes)*kernel_num

        x = self.dropout(x)
        logits = self.fc(x)
        # print(logits)
        return F.log_softmax(logits)

        return logits

class CNN2(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, model_config):
        super(CNN2, self).__init__(general_config)

        self.nembedding = 300 # model_config['embedding_dim']
        kernel_num = 3
        kernel_sizes = [3, 4, 5]
        label_size = label_size
        dropout = 0.3
        use_pretrain = False
        embed_matrix = None
        embed_freeze = False

        self.embedding = nn.Embedding(vocab_size, self.nembedding)
        if use_pretrain is True:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embed_matrix).type(torch.FloatTensor),
                                                 requires_grad=not embed_freeze)
        self.in_channel = 1
        self.out_channel = kernel_num
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.convs1 = nn.ModuleList([nn.Conv2d(self.in_channel, self.out_channel, (K, self.nembedding))
                                     for K in self.kernel_sizes])  # kernel_sizes,like (3,4,5)

        self.dropout = nn.Dropout(dropout)
        """
        in_feature=len(kernel_sizes)*kernel_num,because we concatenate 
        """
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, label_size)

    def forward(self, data):
        sequences, lengths = data

        packed_sequence = pack_padded_sequence(sequences, lengths, batch_first=True)

        padded_sentences, lengths = pad_packed_sequence(packed_sequence, padding_value=int(0),
                                                        batch_first=True)  # set batch_first true
        #padded sequence
        x = self.embedding(padded_sentences)  # batch_size*num_word*nembedding
        # print(x.shape)
        x = x.unsqueeze(1)  # (batch_size,1,num_word,nembedding)   1 is in_channel
        # print("X unzq: {}".format(x.shape))
        # for conv in self.convs1:
        #     print(conv(x))
        # print([F.relu(conv(x)).squeeze(3) for conv in self.convs1])
        # print("THERE3")

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # a list containing (batch_size,out_channel,W)
        # print(x)
        # print("THERE")
        x = [F.max_pool1d(e, e.size(2)).squeeze(2) for e in
             x]  # max_pool1d(input, kernel_size),now x is a list of (batch_size,out_channel)
        # print("HERE")
        x = torch.cat(x, dim=1)  # concatenate , x is batch_size,len(kernel_sizes)*kernel_num

        x = self.dropout(x)
        logits = self.fc(x)
        # print(logits)
        return F.log_softmax(logits)

        return logits

# class StackedCRNN(BaseModel):
#     def __init__(self, vocab_size, label_size, general_config, config):
#         super(StackedCRNN, self).__init__(general_config)
#         cnn_config = config["cnn"]
#         rnn_config = config["rnn"]
#
#         self.embedding = nn.Embedding(
#             vocab_size, config["nembedding"], padding_idx=0)
#         self.convs = nn.ModuleList([
#             nn.Conv2d(1, Nk, Ks) for Ks, Nk in zip(cnn_config["kernel_sizes"],
#                                                    cnn_config["nkernels"])
#         ])
#         self.lstm = nn.LSTM(
#             input_size=cnn_config["nkernels"][-1],
#             hidden_size=rnn_config["nhidden"],
#             num_layers=rnn_config["nlayers"],
#             dropout=rnn_config["dropout"],
#             bidirectional=False)
#         self.dense = nn.Linear(
#             in_features=rnn_config["nhidden"], out_features=label_size)
#
#     def forward(self, entity_ids):#, seq_len):
#         x = self.embedding(entity_ids)
#         #TODO: Understand why the the transpose here...
#         # x = x.transpose(0, 1)
#
#         for i, conv in enumerate(self.convs):
#             # Since we are using conv2d, we need to add extra outer dimension
#             x = x.unsqueeze(1)
#             x = F.relu(conv(x)).squeeze(3)
#             x = x.transpose(1, 2)
#
#         out, _ = self.lstm(x.transpose(0, 1))
#         last_output = out[-1, :, :]
#         logits = self.dense(last_output)
#         return F.log_softmax(logits)
#
#         return logits

class StackedCRNN(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, config):
        super(StackedCRNN, self).__init__(general_config)
        cnn_config = config["cnn"]
        rnn_config = config["rnn"]

        self.embedding = nn.Embedding(
            vocab_size, config["nembedding"], padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, Nk, Ks) for Ks, Nk in zip(cnn_config["kernel_sizes"],
                                                   cnn_config["nkernels"])
        ])
        self.lstm = nn.LSTM(
            input_size=cnn_config["nkernels"][-1],
            hidden_size=rnn_config["nhidden"],
            num_layers=rnn_config["nlayers"],
            dropout=rnn_config["dropout"],
            bidirectional=False)
        self.dense = nn.Linear(
            in_features=rnn_config["nhidden"], out_features=label_size)

    def forward(self, data):#, seq_len):

        sequences, lengths = data

        packed_sequence = pack_padded_sequence(sequences, lengths, batch_first=True)

        padded_sentences, lengths = pad_packed_sequence(packed_sequence, padding_value=int(0),
                                                        batch_first=True)

        x = self.embedding(padded_sentences)
        #TODO: Understand why the the transpose here...
        # x = x.transpose(0, 1)

        for i, conv in enumerate(self.convs):
            # Since we are using conv2d, we need to add extra outer dimension
            x = x.unsqueeze(1)
            x = F.relu(conv(x)).squeeze(3)
            x = x.transpose(1, 2)

        out, _ = self.lstm(x.transpose(0, 1))
        last_output = out[-1, :, :]
        logits = self.dense(last_output)
        return F.log_softmax(logits)

        return logits


#multinomial naiveBayes
class NaiveBayes(BaseModel):
    def __init__(self, classes , general_config, config,preprocessing_config):
        super(NaiveBayes, self).__init__(config)
        self.general_config = general_config
        self.config = config
        self.clf = MultinomialNB()
        self.classes = classes

    def partial_fit(self, X_train, y_train):
        self.clf.partial_fit(X_train,y_train, classes=self.classes)

    def predict(self, X_test):
        return self.clf.predict(X_test)

from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim
import numpy as np
import h5py
import logging
from tqdm import tqdm
from numpy import pi
from sklearn.naive_bayes import MultinomialNB

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

class LSTM2(BaseModel):



    def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTM2, self).__init__(general_config)

        self.batch_size = general_config['data_loader']['batch_size']

        self.hidden_dim = model_config['hidden_dim']

        #TODO: remove this
        self.use_gpu = True

        if 'word2vec' in general_config:
            w2v_file = general_config['word2vec_files'][general_config['word2vec']]
            logging.info("Loading w2v file: {}".format(w2v_file))
            w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file,
                                                                          binary=True,
                                                                          unicode_errors='ignore')
            embed_dim = w2v.vector_size

            vocab_file = h5py.File(general_config['vocab_file'], 'r', libver='latest', swmr=True)

            self.word_embeddings =  nn.Embedding(vocab_size, embed_dim)
            logging.info("Updating vocab vectors...")
            random_vectors_count = 0
            for word in tqdm(vocab_file.keys()):
                index = int(vocab_file[word].value)
                try:
                    self.word_embeddings.weight.data[index,:].set_(torch.FloatTensor(w2v.wv[word]))

                except KeyError:
                    random_vectors_count
                    self.word_embeddings.weight.data[index,:].set_(torch.FloatTensor(np.random.normal(scale=0.6, size=(embed_dim,))))

            self.word_embeddings.weight.requires_grad = False

            if random_vectors_count > 0:
                logging.warning("{:.1f}% words do not exist in the pretrained embeddings".format(float(random_vectors_count)/len(vocab_file)))
        else:
            embed_dim = model_config['embedding_dim']
            self.word_embeddings =  nn.Embedding(vocab_size,embed_dim)

        # self.word_embeddings, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix, True)


        # self.word_embeddings = nn.Embedding.from_pretrained(weights)

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

    def create_emb_layer(self, weights_matrix, non_trainable=False):

        num_embeddings, embedding_dim = weights_matrix.shape#.size()

        emb_layer = nn.Embedding(num_embeddings, embedding_dim)

        emb_layer.load_state_dict({'weight': weights_matrix})

        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

class LSTM2Layer(BaseModel):



    def __init__(self, vocab_size, label_size, general_config, model_config): #embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTM2Layer, self).__init__(general_config)

        self.batch_size = general_config['data_loader']['batch_size']

        self.hidden_dim = model_config['hidden_dim']

        #TODO: remove this
        self.use_gpu = True

        if 'word2vec' in general_config:
            w2v_file = general_config['word2vec_files'][general_config['word2vec']]
            logging.info("Loading w2v file: {}".format(w2v_file))
            w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file,
                                                                          binary=True,
                                                                          unicode_errors='ignore')
            embed_dim = w2v.vector_size

            vocab_file = h5py.File(general_config['vocab_file'], 'r', libver='latest', swmr=True)

            self.word_embeddings =  nn.Embedding(vocab_size, embed_dim)
            logging.info("Updating vocab vectors...")
            random_vectors_count = 0
            for word in tqdm(vocab_file.keys()):
                index = int(vocab_file[word].value)
                try:
                    self.word_embeddings.weight.data[index,:].set_(torch.FloatTensor(w2v.wv[word]))

                except KeyError:
                    random_vectors_count
                    self.word_embeddings.weight.data[index,:].set_(torch.FloatTensor(np.random.normal(scale=0.6, size=(embed_dim,))))

            self.word_embeddings.weight.requires_grad = False

            if random_vectors_count > 0:
                logging.warning("{:.1f}% words do not exist in the pretrained embeddings".format(float(random_vectors_count)/len(vocab_file)))
        else:
            embed_dim = model_config['embedding_dim']
            self.word_embeddings =  nn.Embedding(vocab_size,embed_dim)

        # self.word_embeddings, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix, True)


        # self.word_embeddings = nn.Embedding.from_pretrained(weights)

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


        # print(self.hidden[0].shape)

        logits = self.hidden2label(self.hidden[0]).squeeze()

        return F.log_softmax(logits)

    def create_emb_layer(self, weights_matrix, non_trainable=False):

        num_embeddings, embedding_dim = weights_matrix.shape#.size()

        emb_layer = nn.Embedding(num_embeddings, embedding_dim)

        emb_layer.load_state_dict({'weight': weights_matrix})

        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

class SVM(BaseModel):
    def __init__(self, vocab_size, label_size, general_config, model_config):
        super(SVM, self).__init__(general_config)
        self.config = general_config
        self.l1 = nn.Linear(vocab_size, label_size)

    def forward(self, x):
        #TODO: Pass thru embedding here!
        y = self.l1(x.float())
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
    def __init__(self, classes , general_config, config):
        super(NaiveBayes, self).__init__(config)
        self.general_config = general_config
        self.config = config
        self.clf = MultinomialNB()
        self.classes = classes

    def partial_fit(self, X_train, y_train):
        self.clf.partial_fit(X_train,y_train, classes=self.classes)

    def predict(self, X_test):
        return self.clf.predict(X_test)

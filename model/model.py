from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class MnistModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)
        self.config = config
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SVM(BaseModel):
    def __init__(self, vocab_size, label_size, config):
        super(SVM, self).__init__(config)
        self.config = config
        self.l1 = nn.Linear(vocab_size, label_size)

    def forward(self, x):
        return self.l1(x)
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)

class StackedCRNNClassifier(BaseModel):
    def __init__(self, vocab_size, label_size, config):
        super(StackedCRNNClassifier, self).__init__(config)
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

    def forward(self, entity_ids, seq_len):
        x = self.embedding(entity_ids)
        x = x.transpose(0, 1)

        for i, conv in enumerate(self.convs):
            # Since we are using conv2d, we need to add extra outer dimension
            x = x.unsqueeze(1)
            x = F.relu(conv(x)).squeeze(3)
            x = x.transpose(1, 2)

        out, _ = self.lstm(x.transpose(0, 1))
        last_output = out[-1, :, :]
        logits = self.dense(last_output)

        return logits
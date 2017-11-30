
# coding: utf-8

# # Recurrent Neural Networks with ``gluon``
# 

import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn


# ## Define classes for indexing words of the input document
# 

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


# The ``Dictionary`` class is used by the ``Corpus`` class to index the words of the input document.

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')


# ## Provide an exposition of different RNN models with ``gluon``
# 

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# ## Select an RNN model and configure parameters
# 


args_data = '../data/nlp/ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 1
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500
args_save = 'model.param'


# ## Load data as batches
# 
# We load the document data by leveraging the aforementioned ``Corpus`` class. 
# 
# To speed up the subsequent data flow in the RNN model, we pre-process the loaded data as batches. This procedure is defined in the following ``batchify`` function.


context = mx.cpu(0)
corpus = Corpus(args_data)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
test_data = batchify(corpus.test, args_batch_size).as_in_context(context)


# ## Build the model
# 
# We go on to build the model, initialize model parameters, and configure the optimization algorithms for training the RNN model.


ntokens = len(corpus.dictionary)

model = RNNModel(args_model, ntokens, args_emsize, args_nhid,
                       args_nlayers, args_dropout, args_tied)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


# ## Train the model and evaluate on validation and testing data sets
# 
# Now we can define functions for training and evaluating the model. The following are two helper functions that will be used during model training and evaluation.



def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


# The following is the function for model evaluation. It returns the loss of the model prediction. We will discuss the details of the loss measure shortly.


def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


# Now we are ready to define the function for training the model. We can monitor the model performance on the training, validation, and testing data sets over iterations.


def train():
    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity %.2f' % (
            epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data)
            model.save_params(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args_lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.load_params(args_save, context)



train()
model.load_params(args_save, context)
test_L = eval(test_data)
print('Best test loss %.2f, test perplexity %.2f'%(test_L, math.exp(test_L)))



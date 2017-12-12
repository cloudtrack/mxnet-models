import time
import numpy as np
import logging
import sys, random, time, math
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import Block, nn
from mxnet import autograd
import _pickle as cPickle
import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile
from six.moves import urllib
from six.moves import xrange  

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = str(f.read(f.namelist()[0]))
  return data
buf = read_data(filename)
vocabulary = buf.split()
def build_dataset(words, n_words):
    dictionary = {}
    reverse_dictionary = ["NA"]
    count = [0] 
    data = [] 
    for word in vocabulary:
        if len(word) == 0:
            continue
        if word not in dictionary:
            dictionary[word] = len(dictionary) + 1
            count.append(0)
            reverse_dictionary.append(word)
        wid = dictionary[word]
        data.append(wid)
        count[wid] += 1
    negative = [] 
    for i, v in enumerate(count):
        if i == 0 or v < 5:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]
    return data, count, dictionary, reverse_dictionary, negative

vocabulary_size = 50000
data, count, dictionary, reverse_dictionary, negative = build_dataset(vocabulary, vocabulary_size)

print(count[2])

class DataBatch(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label

class Word2VecDataIterator(mx.io.DataIter):
    def __init__(self,batch_size=512, negative_samples=5, window=5, num_skips=2):
        super(Word2VecDataIterator, self).__init__()
        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.window = window
        self.num_skips = num_skips
        self.data, self.negative, self.vocab = (data, negative, dictionary)
    @property
    def provide_data(self):
        return [('contexts', (self.batch_size, 1))]

    @property
    def provide_label(self):
        return  [('targets', (self.batch_size, self.negative + 1))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def generate_sample(self, pos, word):
        boundary = self.window
        while(True):
            index = random.randint(-boundary, boundary+1)
            if (index != 0 and pos + boundary >= 0 and pos + boundary < len(self.data)):
                center_word = word
                context_word = self.data[pos + index]
                if center_word != context_word:
                    targets_vec = []
                    targets_vec.append(context_word)
                    while len(targets_vec) < self.negative_samples + 1:
                        w = self.sample_ne()
                        if w != word:
                            targets_vec.append(w)
                    return [word], targets_vec
    def __iter__(self):
        center_data = []
        targets = []
        result = 0
        for pos, word in enumerate(self.data):
            for i in range(self.num_skips):
                center_vec, context_vecs = self.generate_sample(pos ,word)
                center_data.append(center_vec)
                targets.append(context_vecs)
            
            if len(center_data) > self.batch_size:
                data_all = [mx.nd.array(center_data[:self.batch_size])]
                label_all = [mx.nd.array(targets[:self.batch_size])]
                yield DataBatch(data_all, label_all)
                center_data = center_data[self.batch_size:]
                targets = targets[self.batch_size:]


VOCAB_SIZE = len(reverse_dictionary)
BATCH_SIZE = 512
WORD_DIM = 100
NEGATIVE_SAMPLES = 5

ctx = mx.gpu()
data_iterator = Word2VecDataIterator(batch_size=BATCH_SIZE,
                                     negative_samples=NEGATIVE_SAMPLES,
                                     window=5, num_skips=2)   
all_batches = []
all_data = []
counting = 0
for batch in data_iterator:
    all_batches.append(batch)
    if(counting % 500 == 0) :
        print(counting)
        if(counting > 1000 ) :
            break
    counting = counting + 1
cPickle.dump(all_data, open('all_batches.p', 'wb'))


class Model(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            
            
            self.center = nn.Embedding(input_dim=VOCAB_SIZE,
                                       output_dim=WORD_DIM,
                                       weight_initializer=mx.initializer.Uniform(1.0/WORD_DIM))
            
            
            self.target = nn.Embedding(input_dim=VOCAB_SIZE,
                                       output_dim=WORD_DIM,
                                       weight_initializer=mx.initializer.Zero())

    def hybrid_forward(self, F, center, targets, labels):
        """
        Returns the word2vec skipgram with negative sampling network.
        :param F: F is a function space that depends on the type of other inputs. If their type is NDArray, then F will be mxnet.nd otherwise it will be mxnet.sym
        :param center: A symbol/NDArray with dimensions (batch_size, 1). Contains the index of center word for each batch.
        :param targets: A symbol/NDArray with dimensions (batch_size, negative_samples + 1). Contains the indices of 1 target word and `n` negative samples (n=5 in this example)
        :param labels: A symbol/NDArray with dimensions (batch_size, negative_samples + 1). For 5 negative samples, the array for each batch is [1,0,0,0,0,0] i.e. label is 1 for target word and 0 for negative samples
        :return: Return a HybridBlock object
        """
        center_vector = self.center(center)
        target_vectors = self.target(targets)
        pred = F.broadcast_mul(center_vector, target_vectors)
        pred = F.sum(data = pred, axis = 2)
        sigmoid = F.sigmoid(pred)
        loss = F.sum(labels * F.log(sigmoid) + (1 - labels) * F.log(1 - sigmoid), axis=1)
        loss = loss * -1.0 / BATCH_SIZE
        loss_layer = F.MakeLoss(loss)
        return loss_layer

model = Model()
model.initialize(ctx=ctx)
model.hybridize() 

trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate':4,'clip_gradient':5})

labels = nd.zeros((BATCH_SIZE, NEGATIVE_SAMPLES+1), ctx=ctx)
labels[:,0] = 1
start_time = time.time()
epochs = 5
for e in range(epochs):
    moving_loss = 0.
    for i, batch in enumerate(all_batches):
        center_words = batch.data[0].as_in_context(ctx)
        target_words = batch.label[0].as_in_context(ctx)
        with autograd.record():
            loss = model(center_words, target_words, labels)
        loss.backward()
        trainer.step(1, ignore_stale_grad=True)
        
        
        if (i == 0) and (e == 0):
            moving_loss = loss.asnumpy().sum()
        else:
            moving_loss = .99 * moving_loss + .01 * loss.asnumpy().sum()
        if (i + 1) % 50 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))
        if i > 15000:
            break

print("1 epoch took %s seconds" % (time.time() - start_time))
 




import time
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import Block, nn
from mxnet import autograd
import _pickle as cPickle
import math
import os
import random
from tempfile import gettempdir
import zipfile
from six.moves import urllib
from six.moves import xrange

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
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


class DataBatch(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label

class Word2VecDataIterator(mx.io.DataIter):
    def __init__(self, batch_size=512, num_neg_samples=5, window=5):
        super(Word2VecDataIterator, self).__init__()
        self.batch_size = batch_size
        self.negative_samples = num_neg_samples
        self.window = window
        self.data, self.negative, self.dictionary = (data, negative, dictionary)

    @property
    def provide_data(self):
        return [('contexts', (self.batch_size, 1))]

    @property
    def provide_label(self):
        return [('targets', (self.batch_size, self.negative + 1))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def __iter__(self):
        input_data = []
        update_targets = []
        for pos, word in enumerate(self.data):
            for index in range(-self.window, self.window + 1):
                if (index != 0 and pos + index >= 0 and pos + index < len(self.data)):
                    context = self.data[pos + index]
                    if word != context:
                        input_data.append([word])
                        targets = []
                        targets.append(context) # positive sample
                        while len(targets) < self.negative_samples + 1: # negative sample
                            w = self.sample_ne()
                            if w != word:
                                targets.append(w)
                        update_targets.append(targets)

            # Check if batch size is full
            if len(input_data) > self.batch_size:
                batch_inputs = [mx.nd.array(input_data[:self.batch_size])]
                batch_update_targets = [mx.nd.array(update_targets[:self.batch_size])]
                yield DataBatch(batch_inputs, batch_update_targets)
                input_data = input_data[self.batch_size:]
                update_targets = update_targets[self.batch_size:]


dictionary_size = len(reverse_dictionary)
batch_size = 512
num_hiddens = 100
num_negative_samples = 5

ctx = mx.gpu()
data_iterator = Word2VecDataIterator(batch_size=batch_size,
                                     num_neg_samples=num_negative_samples,
                                     window=5, num_skips=2)
batches = []
training_data = []
counting = 0
for batch in data_iterator:
    batches.append(batch)
    if (counting % 500 == 0):
        print(counting)
    counting = counting + 1

    cPickle.dump(training_data, open('all_batches.p', 'wb'))


class Model(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.center = nn.Embedding(input_dim=dictionary_size,
                                       output_dim=num_hiddens,
                                       weight_initializer=mx.initializer.Uniform(1.0 / num_hiddens))

            self.target = nn.Embedding(input_dim=dictionary_size,
                                       output_dim=num_hiddens,
                                       weight_initializer=mx.initializer.Zero())

    def hybrid_forward(self, F, center, targets, labels):
        input_vectors = self.center(center)
        update_targets = self.target(targets)
        predictions = F.broadcast_mul(input_vectors, update_targets)
        predictions = F.sum(data=predictions, axis=2)
        sigmoid = F.sigmoid(predictions)
        loss = F.sum(labels * F.log(sigmoid) + (1 - labels) * F.log(1 - sigmoid), axis=1)
        loss = loss * -1.0 / batch_size
        loss_layer = F.MakeLoss(loss)
        return loss_layer


model = Model()
model.initialize(ctx=ctx)
model.hybridize()

trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 4, 'clip_gradient': 5})

labels = nd.zeros((batch_size, num_negative_samples + 1), ctx=ctx)
labels[:, 0] = 1
start_time = time.time()
num_epochs = 5

def get_loss(epoch_n, batch_n, loss):
    if(batch_n==0 and epoch_n==0):
        loss = loss.asnumpy().sum()
    else:
        loss = .99 * loss + .01 * loss.asnumpy().sum()
    if(i + 1) % 100 == 0:
        print("%sth epoch , %sth batch. avg of loss: %s" % (epoch_n, batch_n, loss))
    return loss

for e in range(num_epochs):
    moving_loss = 0.
    for i, batch in enumerate(batches):
        center_words = batch.data[0].as_in_context(ctx)
        target_words = batch.label[0].as_in_context(ctx)
        with autograd.record():
            loss = model(center_words, target_words, labels)
        loss.backward()
        # ignore_stale_grad ; only update calculated target weights
        trainer.step(1, ignore_stale_grad=True)
        moving_loss = get_loss(e, i, moving_loss)
        if i > 15000:
            break
    print("1 epoch took %s seconds" % (time.time() - start_time))

# format index : vector
key = list(model.collect_params().keys())
all_vecs = model.collect_params()[key[0]].data().asnumpy()
cPickle.dump(all_vecs, open('all_vecs.p', 'wb'))

#  foramt word : vector
w2vec_dict = dictionary.copy()
for word in dictionary:
    idx = dictionary[word]
    vector = all_vecs[idx]
    w2vec_dict[word] = vector

cPickle.dump(w2vec_dict, open('w2vec_dict.p', 'wb'))


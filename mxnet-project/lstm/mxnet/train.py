from pre_processing import *
from sklearn.model_selection import train_test_split
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn, rnn


x_train, x_test, y_train, y_test = train_test_split(train_data, train_all_labels, test_size=0, random_state=42)
# x_train = train_data
# y_train = train_all_labels

x_test = test_data
y_test = test_all_labels

# divide 30% of data into test data
# x_train, x_test, y_train, y_test = train_test_split(train_data, train_all_labels, test_size=0.3, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.5, random_state=42)


# some statistics
min_len = min(map(len, train_data))
max_len = max(map(len, train_data))
avg_len = sum(map(len, train_data)) / len(train_data)

print("min len: ", min_len)
print("max len: ", max_len)
print("avg len: ", avg_len)

seq_len = 250 # set the max word length of each movie review

# if sentence is greater than max_len, truncates
# if less, pad with value
def pad_sequences(sentences, max_len=500, value = 0):
    padded_sentences = []
    for sentence in sentences:
        new_sentence = []
        if (len(sentence) > max_len):
            new_sentence = sentence[:max_len]
            padded_sentences.append(new_sentence)
        else:
            new_sentence = np.append(sentence, [value]*(max_len-len(sentence)))
            padded_sentences.append(new_sentence)

    return padded_sentences

context = mx.gpu()
X_train = nd.array(pad_sequences(x_train, max_len=seq_len, value=0), context)
X_test = nd.array(pad_sequences(x_test, max_len=seq_len, value=0), context)
Y_train = nd.array(y_train, context)
Y_test = nd.array(y_test, context)

print("X_train: " + str(X_train))
print("X_test: " + str(X_test))

print("Y_train: " + str(Y_train))
print("Y_test: " + str(Y_test))



## define network
num_classes = 2
num_hidden = 200
learning_rate = .01
epochs = 200
batch_size = 20

model = nn.Sequential()
with model.name_scope():
    model.embed = nn.Embedding(voca_size, num_embed)
    model.add(rnn.LSTM(num_hidden, layout = 'NTC', dropout=0.7, bidirectional=False))
    model.add(nn.Dense(num_classes))

def eval_accuracy(x, y, batch_size):
    accuracy = mx.metric.Accuracy()

    for i in range(x.shape[0] // batch_size):
        data = x[i*batch_size:(i*batch_size + batch_size), ]
        target = y[i*batch_size:(i*batch_size + batch_size), ]

        output = model(data)
        predictions = nd.argmax(output, axis=1)
        accuracy.update(preds=predictions, labels=target)

    return accuracy.get()[1]

model.collect_params().initialize(mx.init.Xavier(), ctx=context)

model.embed.weight.set_data(embedding_matrix.as_in_context(context))

trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': learning_rate})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

for epoch in range(epochs):

    for b in range(X_train.shape[0] // batch_size):
        data = X_train[b*batch_size:(b*batch_size + batch_size),]
        target = Y_train[b*batch_size:(b*batch_size + batch_size),]

        data = data.as_in_context(context)
        target = target.as_in_context(context)

        with autograd.record():
            output = model(data)
            L = softmax_cross_entropy(output, target)
        L.backward()
        trainer.step(data.shape[0])

    test_accuracy = eval_accuracy(X_test, Y_test, batch_size)
    train_accuracy = eval_accuracy(X_train, Y_train, batch_size)
    print("Epoch %s. Train_acc %s, Test_acc %s" %
          (epoch, train_accuracy, test_accuracy))

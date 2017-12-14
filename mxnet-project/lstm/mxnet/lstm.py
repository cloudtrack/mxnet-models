
import os
import io
import re
import _pickle as pkl
import numpy as np
import mxnet as mx

from collections import Counter
from mxnet import nd
from sklearn.model_selection import train_test_split
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn, rnn

def make_dictionary(dictionary_file_name, sentiments):
    # if file exist, return file
    if os.path.exists(dictionary_file_name):
        f = open(dictionary_file_name, 'rb')
        word_dict = pkl.load(f)
        f.close()
        return word_dict

    else:
        # count word
        word_counter = count_word(sentiments)
        word_dict = create_word_index(word_counter)
        save_file(word_dict, dictionary_file_name)
        return word_dict

def save_file(file_obj, file_name):
    f = open(file_name, 'wb')
    pkl.dump(file_obj, f, -1)
    f.close()

def create_word_index(word_counter):
    idx = 1
    word_dict = {}

    for word in word_counter.most_common():
        word_dict[word[0]] = idx
        idx += 1

    return word_dict 


def count_word(sentiments):
    word_counter = Counter()
    for sentiment in sentiments:
        for word in (clear_str(sentiment)).split():
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1

    return word_counter

def clear_str(sentiment):
    string = sentiment.lower().replace("<br />", " ")
    remove_spe_chars = re.compile("[^A-Za-z0-9 ]+")

    return re.sub(remove_spe_chars, "", string.lower())
	

def read_files(folder_name):
    sentiments =[]
    filenames = os.listdir(os.curdir+"/"+folder_name)

    for file in filenames:
        with open(folder_name+"/"+file, "r", encoding="utf8") as f:
            data = f.read().replace("\n", "")
            sentiments.append(data)

    return sentiments

# prepare train data
data_path = '../data/aclImdb'

train_pos_foldername = data_path + "/train/pos"
train_pos_sentiments = read_files(train_pos_foldername)

train_neg_foldername = data_path + "/train/neg"
train_neg_sentiments = read_files(train_neg_foldername)

train_pos_labels = [1 for _ in train_pos_sentiments]
train_neg_labels = [0 for _ in train_neg_sentiments]
train_all_labels = train_pos_labels + train_neg_labels

train_all_sentiments = train_pos_sentiments + train_neg_sentiments

#prepare test data
test_pos_foldername = data_path + "/test/pos"
test_pos_sentiments = read_files(test_pos_foldername)

test_neg_foldername = data_path + "/test/neg"
test_neg_sentiments = read_files(test_neg_foldername)

test_pos_labels = [1 for _ in test_pos_sentiments]
test_neg_labels = [0 for _ in test_neg_sentiments]
test_all_labels = test_pos_labels + test_neg_labels

dictionary_file_name = 'imdb.dict.pkl'
word_dict = make_dictionary( dictionary_file_name, train_all_sentiments )


def encode_sentences(input_file, word_dict):
    output_string = []
    for line in input_file:
        output_line = []
        for word in clear_str(line).split():
            if word in word_dict:
                output_line.append(word_dict[word])
        output_string.append(output_line)

    return output_string

train_pos_encoded = encode_sentences(train_pos_sentiments, word_dict)
train_neg_encoded = encode_sentences(train_neg_sentiments, word_dict)
train_all_encoded = train_pos_encoded + train_neg_encoded

test_pos_encoded = encode_sentences(test_pos_sentiments, word_dict)
test_neg_encoded = encode_sentences(test_neg_sentiments, word_dict)
test_all_encoded = test_pos_encoded + test_neg_encoded

voca_size = 10000 # total number of voca we will track
train_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in train_all_encoded]
test_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in test_all_encoded]

## word2vec precess
num_embed = 300 # richness of the word attributes captured

def load_glove_index(path):
    print("load_glove_index...")
    import io
    f = io.open(path, encoding="utf8")
    embeddings_index = {}

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index



def create_embed(filename, glove_path):
    if os.path.exists(filename):
        f = open(filename, 'rb')
        embedding_matrix = pkl.load(f)
        f.close()
        return embedding_matrix

    embedding_index = load_glove_index(glove_path)
    
    embedding_matrix = np.zeros((voca_size, num_embed))
    for word, i in word_dict.items():
        if i >= voca_size:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_matrix = nd.array(embedding_matrix)
    save_file(embedding_matrix, filename)
    return embedding_matrix


embed_matrix_filename = 'embed.metrix.pkl'
glove_path = '../data/glove.42B.300d.txt'
embedding_matrix = create_embed(embed_matrix_filename, glove_path)


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
num_hidden = 25
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

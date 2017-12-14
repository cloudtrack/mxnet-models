import re
import itertools
from collections import Counter
import numpy as np
import _pickle as pkl

from mxnet import nd

## read data
def read_files(foldername):
    import os
    import io
    sentiments = []
    filenames = os.listdir(os.curdir+"/"+foldername)

    for file in filenames:
        with io.open(foldername + "/" + file, "r", encoding="utf8") as open_file:
            data = open_file.read().replace('\n', '')
            sentiments.append(data)

    return sentiments

## prepare train data
# pos_foldername = "../data/aclImdb/train/pos/dev"
train_pos_foldername = "../data/aclImdb/train/pos"
train_pos_sentiments = read_files(train_pos_foldername)

# neg_foldername = "../data/aclImdb/train/neg/dev"
train_neg_foldername = "../data/aclImdb/train/neg"
train_neg_sentiments = read_files(train_neg_foldername)

train_all_sentiments = train_pos_sentiments + train_neg_sentiments

train_pos_labels = [1 for _ in train_pos_sentiments]
train_neg_labels = [0 for _ in train_neg_sentiments]
train_all_labels = train_pos_labels + train_neg_labels

## prepare test_data
test_pos_foldername = "../data/aclImdb/test/pos"
test_pos_sentiments = read_files(test_pos_foldername)

# neg_foldername = "../data/aclImdb/train/neg/dev"
test_neg_foldername = "../data/aclImdb/test/neg"
test_neg_sentiments = read_files(test_neg_foldername)

test_all_sentiments = test_pos_sentiments + test_neg_sentiments
all_sentiments = train_all_sentiments + test_all_sentiments

test_pos_labels = [1 for _ in test_pos_sentiments]
test_neg_labels = [0 for _ in test_neg_sentiments]
test_all_labels = test_pos_labels + test_neg_labels

all_labels = train_all_labels + test_all_labels

## make dictionary..
def clear_str(string):
    string = string.lower().replace("<br />", " ")
    remove_spe_chars = re.compile("[^A-Za-z0-9 ]+")

    return re.sub(remove_spe_chars, "", string.lower())

word_counter = Counter()

# make whole word count
def create_count(sentiments):
    print("create_count...")
    for sentiment in sentiments:
        for word in (clear_str(sentiment)).split():
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1

# assign idx to words
def create_word_index():
    idx = 1
    word_dict = {}

    for word in word_counter.most_common():
        word_dict[word[0]] = idx
        idx += 1

    return word_dict

# create_count( train_all_sentiments )
create_count( all_sentiments )
word_dict = create_word_index()

# save dictionary
f = open('imdb.dict.pkl', 'wb')
pkl.dump(word_dict, f, -1)
f.close()

# open dictionary
f = open('imdb.dict.pkl', 'rb')
word_dict = pkl.load(f)
f.close()

idx2word = { v: k for k, v in word_dict.items()}

## helper function: encode sentences
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

all_encoded = train_all_encoded + test_all_encoded

voca_size = 10000 # total number of voca we will track
train_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in train_all_encoded]
test_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in test_all_encoded]
# all_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in all_encoded]
all_data = train_data + test_data


## word2vec precess
num_embed = 300 # richness of the word attributes captured

def load_glove_index(loc):
    print("load_glove_index...")
    import io
    f = io.open(loc, encoding="utf8")
    embeddings_index = {}

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def create_emb():
    print("create_emb...")
    embedding_matrix = np.zeros((voca_size, num_embed))
    for word, i in word_dict.items():
        if i >= voca_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_matrix = nd.array(embedding_matrix)
    return embedding_matrix

embeddings_index = load_glove_index('../data/glove.42B.300d.txt')
embedding_matrix = create_emb()

import re
import itertools
from collections import Counter
import numpy as np
import cPickle as pkl

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

pos_foldername = "../data/aclImdb/train/pos/dev"
pos_sentiments = read_files(pos_foldername)

neg_foldername = "../data/aclImdb/train/neg/dev"
neg_sentiments = read_files(neg_foldername)

pos_labels = [1 for _ in pos_sentiments]
neg_labels = [0 for _ in neg_sentiments]

## make dictionary..
def clear_str(string):
    string = string.lower().replace("<br />", " ")
    remove_spe_chars = re.compile("[^A-Za-z0-9 ]+")

    return re.sub(remove_spe_chars, "", string.lower())

word_counter = Counter()

# make whole word count
def create_count(sentiments):
    print("create_count...")
    idx = 0;
    for sentiment in sentiments:
        for word in (clear_str(sentiment)).split():
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1

            idx += 1;
            if ( idx % 10000 == 0 ):
                print("idx: " + str(idx))

# assign idx to words
def create_word_index():
    idx = 1
    word_dict = {}

    for word in word_counter.most_common():
        word_dict[word[0]] = idx
        idx += 1

    return word_dict

create_count( pos_sentiments + neg_sentiments )
word_dict = create_word_index()

# save dictionary
f = open('imdb.dict.pkl', 'wb')
pkl.dump(word_dict, f, -1)
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

pos_encoded = encode_sentences(pos_sentiments, word_dict)
neg_encoded = encode_sentences(neg_sentiments, word_dict)
all_encoded = pos_encoded + neg_encoded


voca_size = 5000 # total number of voca we will track
t_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in all_encoded]

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

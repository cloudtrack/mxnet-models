import os
import re
import numpy as np
import mxnet as mx 
import _pickle as pkl

from mxnet import nd, gluon
from mxnet.gluon import nn, rnn

def read_files(folder_name):
    sentiments =[]
    filenames = os.listdir(os.curdir+"/"+folder_name)

    for file in filenames:
        with open(folder_name+"/"+file, "r", encoding="utf8") as f:
            data = f.read().replace("\n", "")
            sentiments.append(data)

    return sentiments

def encode_sentences(input_file, word_dict):
    # print("input file: " + str( input_file ))
    output_string = []
    for line in input_file:
        # print("line: " +str(line))
        output_line = []
        for word in clear_str(line).split():
            if word in word_dict:
                output_line.append(word_dict[word])
        output_string.append(output_line)

    return output_string

def get_dictionary(dictionary_file_name):
    # if file exist, return file
    if os.path.exists(dictionary_file_name):
        f = open(dictionary_file_name, 'rb')
        word_dict = pkl.load(f)
        f.close()
        return word_dict

    else:
        print( "Error: word dictionary is not exist!" )
        exit(1)

def clear_str(sentiment):
    string = sentiment.lower().replace("<br />", " ")
    remove_spe_chars = re.compile("[^A-Za-z0-9 ]+")

    return re.sub(remove_spe_chars, "", string.lower())

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

def predict_new_review(review, word_dict, model):
    # print("review: " +str(review))
    test_encoded = encode_sentences(review, word_dict)

    # print("encoded: " +str(test_encoded))
 
    voca_size = 10000 
    test_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in test_encoded]

    seq_len = 250
    context = mx.gpu()
    X_test = nd.array(pad_sequences(test_data, max_len=seq_len, value=0), context)   

    # print( "X_test:" + str(X_test) )

    output = model(X_test)
    # print( "output: " + str(output) )
    predict = nd.argmax(output, axis = 1)
    # print( "predict: " + str(predict) )

    if ( predict[0] == 0 ):
        print("result: negative\n")
    else:
        print("result: positive\n")


def show_prepared_test(word_dict, model):
    testfile_path = "../data/aclImdb/demo"
    testfiles = read_files(testfile_path)

    for data in testfiles:
        print( "\ntest review: " + str(data) )
        data = [data]
        test_encoded = encode_sentences(data, word_dict)
  
        voca_size = 10000 
        test_data = [np.array([i if i < (voca_size-1) else (voca_size-1) for i in s]) for s in test_encoded]

        seq_len = 250
        context = mx.gpu()

        X_test = nd.array(pad_sequences(test_data, max_len=seq_len, value=0), context)   

        output = model(X_test)
        predict = nd.argmax(output, axis = 1)

        if ( predict[0] == 0 ):
            print("result: negative\n")
        else:
            print("result: positive\n")

# callback model
def load_model():
    num_classes = 2
    num_hidden = 25
    num_embed = 300
    learning_rate = .01
    epochs = 200
    batch_size = 20
    voca_size = 10000
    context = mx.gpu()

    model_params_filename = "lstm_net.params_epoch4"
    model = nn.Sequential()
    with model.name_scope():
        model.embed = nn.Embedding(voca_size, num_embed)
        model.add(rnn.LSTM(num_hidden, layout = 'NTC', dropout=0.7, bidirectional=False))
        model.add(nn.Dense(num_classes))

    model.load_params(model_params_filename, context)
    return model

if __name__ == "__main__":
    # load model
    print("load the pre-trained LSTM model...")
    model = load_model()

    # load dictionary
    dict_filename = "imdb.dict.pkl"
    word_dict = get_dictionary(dict_filename)

    # get input
    while( True ):
        print("\nType movie review.")
        print("If you type 'quit', this program terminates.")
        print("If you type 'prepared test', program shows prepared test set.")
        review = input(":\n")
        if review == "quit":
            print( "program ends." )
            quit()

        elif review == "prepared test":
            show_prepared_test(word_dict, model)
        else:
            review_list = []
            review_list.append(review)
            predict_new_review(review_list, word_dict, model)
            

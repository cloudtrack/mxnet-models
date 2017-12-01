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


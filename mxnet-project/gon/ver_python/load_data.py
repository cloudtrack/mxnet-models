# -*- coding: utf-8 -*-

import cPickle
import gzip
import os

import numpy

def load_data(dataset='mnist.pkl.gz'):
  dataset = os.path.join(os.path.split(__file__)[0], '../data', dataset)
  f = gzip.open(dataset, 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()

  def make_numpy_array(data_xy):
    data_x, data_y = data_xy
    return numpy.array(data_x), numpy.array(data_y)

  train_set_x, train_set_y = make_numpy_array(train_set)
  valid_set_x, valid_set_y = make_numpy_array(valid_set)
  test_set_x, test_set_y = make_numpy_array(test_set)

  rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
          (test_set_x, test_set_y)]

  return rval
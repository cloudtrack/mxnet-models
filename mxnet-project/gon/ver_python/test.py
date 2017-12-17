# -*- coding: utf-8 -*-

import time

from rbm2 import RBM
from load_data import load_data
import mxnet as mx
def test(learning_rate=0.1, k=1, training_epochs=15):
  print '... loading data'

  datasets = load_data('mnist.pkl.gz')
  train_set_x, train_set_y = datasets[0]
  test_set_x, test_set_y = datasets[2]



  print '... modeling'

  rbm = RBM(input=train_set_x, n_visible=28 * 28, n_hidden= 28* 28)

  print '... training'

  start_time = time.clock()

  for epoch in xrange(training_epochs):
    cost = rbm.get_cost_updates(lr=learning_rate, k=k)
    print 'Training epoch %d, cost is ' % epoch, cost

  end_time = time.clock()
  pretraining_time = (end_time - start_time)

  print ('Training took %f minutes' % (pretraining_time / 60.))

if __name__ == '__main__':
  test()
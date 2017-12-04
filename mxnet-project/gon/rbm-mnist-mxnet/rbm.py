# -*- coding: utf-8 -*-
import mxnet as mx

#from __future__ import print_function

#import timeit

#try:
#    import PIL.Image as Image
#except ImpimortError:
#    import Image

import numpy as np

#import theano
#import theano.tensor as T
#import os

#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from utils import tile_raster_images
#from logistic_sgd import load_data

class RBM(object):
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if W is None:
            W = mx.symbol.var('W', shape=(n_visible, n_hidden), init=mx.init.Xavier())

        if hbias is None:
            hbias = mx.symbol.var('hbias', shape=(n_hidden), init=mx.init.Constant(0))

        if vbias is None:
            vbias = mx.symbol.var('vbias', shape=(n_visible), init=mx.init.Constant(0))
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        self.params = [self.W, self.hbias, self.vbias]
        print(self.W, self.hbias, self.vbias)

    def free_energy(self, v_sample):
        wx_b = mx.sym.dot(v_sample, self.W) + self.hbias
        vbias_term = mx.sym.dot(v_sample, self.vbias)
        hidden_term = mx.sym.sum(mx.sym.log(1 + mx.sym.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        print(vis)
        print(self.W)
        pre_sigmoid_activation = mx.sym.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, mx.symbol.Activation(pre_sigmoid_activation, 'sigmoid')]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = mx.symbol.random_negative_binomial(shape=(self.n_hidden),
                                             k=1, p=h1_mean)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = mx.sym.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, mx.symbol.Activation(pre_sigmoid_activation, 'sigmoid')]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        v1_sample = mx.symbol.random_negative_binomial(shape=(self.v_hidden),
                                             k=1, p=v1_mean)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, k=1):
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = ph_sample
        #theano scan
        #k times
        (
            pre_sigmoid_nvs,
            nv_means,
            nv_samples,
            pre_sigmoid_nhs,
            nh_means,
            nh_samples
        ) = self.gibbs_hvh(
                self.gibbs_hvh(
                    self.gibbs_hvh(
                        self.gibbs_hvh(
                            chain_start
                        ).v1_sample).v1_sample).v1_sample).v1_sample

        chain_end = nv_samples

        cost = mx.sym.mean(self.free_energy(self.input)) - mx.sym   .mean(
            self.free_energy(chain_end))
        return cost
        """
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
            # reconstruction cross-entropy is a better proxy for CD
        monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        """

def test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500):
    mnist = mx.test_utils.get_mnist()
    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    data = mx.sym.var('data')
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.flatten(data=data)

    >> > y = e.forward()
    >> > y
    print('test')
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    """
#    datasets = load_data(dataset)

#    train_set_x, train_set_y = datasets[0]
#    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
#    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
#    index = T.lscalar()    # index to a [mini]batch
#    x = T.matrix('x')  # the data is presented as rasterized images

#    rng = numpy.random.RandomState(123)
#    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
#    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
#                                                 dtype=theano.config.floatX),
#                                     borrow=True)

    # construct the RBM class

    print(data)
    rbm = RBM(input=data, n_visible=28 * 28,
              n_hidden=n_hidden)

    cost = rbm.get_cost_updates(lr=learning_rate, k=15)


test_rbm(learning_rate=0.1, training_epochs=15,
             dataset='mnist.pkl.gz', batch_size=20,
             n_chains=20, n_samples=10, output_folder='rbm_plots',
             n_hidden=500)
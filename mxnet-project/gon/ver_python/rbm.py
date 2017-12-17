# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
import copy


class RBM(object):

    def __init__(self, input, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.w = mx.symbol.Variable('weight')
        self.h_bias = mx.symbol.Variable('h_bias')
        self.v_bias = mx.symbol.Variable('v_bias')
        self.input = input
        self.var_input = mx.symbol.Variable('var_input')
        weight = self.initialize_weight(self.n_visible, self.n_hidden)
        print(weight)
        self.w.bind(mx.cpu(), {mx.nd.array(weight)})
        self.rng = np.random.RandomState(1234)
        # self.h_samples_ = mx.nd.array(np.zeros((self.batch_size, (n_visibile, n_hidden))))
        self.h_samples_ = mx.symbol.Variable('h_samples_')


    def initialize_weight(self, n_visible, n_hidden):
        w = np.asarray( np.random.RandomState(1234).uniform(
                          low=-4 * np.sqrt(6. / (n_visible + n_hidden)),
                          high=4 * np.sqrt(6. / (n_visible + n_hidden)),
                          size= (n_visible, n_hidden))
                        ,dtype=np.dtype(np.float32))
        return w

    def propup(self, vis):
        wx_b = mx.symbol.FullyConnected(input = vis, weight = self.w,
                                                    bias = self.h_bias, n_hidden = self.n_hidden)
        sigmoid_activation = mx.symbol.Activation(input=wx_b, act_type = 'sigmoid', name = 'activation')
        return sigmoid_activation

    def p_vh(self, hid):
        # hid = self.prop_up()
        w_t = mx.symbol.transpose(input = self.w)
        wx_b = mx.symbol.FullyConnected(input = hid, weight= w_t,
                                        bias=self.v_bias, n_hidden=self.n_hidden)
        p_vh = mx.symbol.Activation(input=wx_b, act_type = 'sigmoid')
        return p_vh

# propup or p_vh is 상호배타적
    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.binomial_random(h1_mean)
        return [h1_mean, h1_sample]

    def propdown(self, hid):
        w_t = mx.symbol.transpose(input=self.w)
        wx_b = mx.symbol.FullyConnected(input = self.hid, weight = w_t,
                                                    bias = self.v_bias, n_hidden = self.n_visible)
        sigmoid_activation = mx.symbol.Activation(input=wx_b, act_type = 'sigmoid', name = 'activation')
        return sigmoid_activation

    def p_hv(self, v):
        'v should be symbol'
        wx_b = mx.symbol.FullyConnected(input=v, weight=self.w,
                                        bias=self.h_bias, n_hidden=self.n_hidden)
        p_hv = mx.symbol.Activation(input=wx_b, act_type='sigmoid')
        return p_hv

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.binomial_random(v1_mean)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_vs(v1_sample)
        return [v1_mean, v1_sample,
                h1_mean, h1_sample]

    def gibbs_vhv(self, v):
        h_sam = self.sample_h_given_vs(v)
        v_sam = self.sample_v_given_h(h_sam)
        return v_sam

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = mx.symbol.FullyConnected(input=v_sample, weight = self.w,
                                        bias=self.h_bias, n_hidden=self.n_hidden)
        vbias_term = mx.symbol.FullyConnected(input=v_sample, weight = self.v_bias,
                                        bo_bias = True, n_hidden=self.n_hidden)
        hidden_term = mx.symbol.sum(input = mx.symbol.log(1+ mx.symbol.exp(input = wx_b)))
        #problem here: parameter of exp should be ndarray

        return -hidden_term - vbias_term

#    def get_cost_updates(self, lr=0.1, k=1):
#         # compute positive phase
#         h_sam = self.p_hv(self.var_input)
#
#         # decide how to initialize persistent chain:
#         # for CD, we use the newly generate hidden sample
#         # for PCD, we initialize from the old state of the chain
#         h_gib = self.gibbs_hvh(h_sam)
#         cost = mx.symbol.mean(input = self.free_energy(self.var_input)) - \
#                mx.symbol.mean(input=self.free_energy(h_gib))

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
      ph_mean, ph_sample = self.sample_h_given_v(self.input)

      if persistent is None:
        chain_start = ph_sample
      else:
        chain_start = persistent

      for step in xrange(k):
        if step == 0:
          nv_means, nv_samples, \
          nh_means, nh_samples = self.gibbs_hvh(chain_start)
        else:
          nv_means, nv_samples, \
          nh_means, nh_samples = self.gibbs_hvh(nh_samples)

      self.W += lr * (np.dot(self.input.T, ph_mean)
                      - np.dot(nv_samples.T, nh_means))
      self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
      self.hbias += lr * np.mean(ph_mean - nh_means, axis=0)

      monitoring_cost = np.mean(np.square(self.input - nv_means))

      return monitoring_cost


    def binomial_random(self, mean):
        sample_array = mx.nd.array(self.rng.random_sample(size=mean.shape))
        sample = mx.sym.Variable('sample')
        sample.bind(ctx=mx.cpu(), args= sample_array)
        p_sample = sample<sample_array
        return p_sample



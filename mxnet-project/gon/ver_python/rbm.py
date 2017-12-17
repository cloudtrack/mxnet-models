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
        self.h_bias_np = self.h_bias.bind(mx.cpu(), {'h_bias': mx.nd.array(np.zeros(n_hidden))}).forward()[0]
        self.v_bias = mx.symbol.Variable('v_bias')
        self.v_bias.bind(mx.cpu(), {'v_bias': mx.nd.array(np.zeros(n_visible))})
        self.input = input
        self.var_input = mx.symbol.Variable('var_input')
        weight = self.initialize_weight(self.n_visible, self.n_hidden)
        self.w_np = self.w.bind(mx.cpu(), {'weight': mx.nd.array(weight)}).forward()[0]
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
        vis_symbol = mx.symbol.Variable('vis')
        vis_symbol.bind(mx.cpu(), {'vis': mx.nd.array(vis)})
        wx_b = mx.symbol.FullyConnected(data=vis_symbol, weight=self.w,
                                        bias=self.h_bias, num_hidden=self.n_visible)
        p_hv = mx.symbol.Activation(data=wx_b, act_type='sigmoid')
        executor = p_hv.bind(mx.cpu(), {'vis': mx.nd.array(vis), 'weight': self.w_np, 'h_bias': self.h_bias_np})
        # get the numpy to make use of np binomial
        print(executor.forward())
        return executor.forward()[0].asnumpy()

    def sample_h_given_v(self, v0_sample):
        # v0_sample is ndarray
        h1_mean = self.propup(v0_sample)
        print(h1_mean)
        #h1_sample = mx.symbol.random_negative_binomial(k=1, p=h1_mean, shape=self.n_hidden)
        print(h1_mean.shape)
        h1_sample = self.rng.binomial(size=h1_mean.size, n=1, p=h1_mean)
        print("sample1 done")
        print(h1_sample)
        return [h1_mean, h1_sample]

    def propdown(self, hid):
        hid_symbol = mx.symbol.Variable('hid')
        executor = hid_symbol.bind(ctx=mx.cpu(), args={'hid': mx.nd.array(hid)})
        print("median")
        print(executor.forward()[0])
        w_t = mx.symbol.transpose(data = self.w)
        wx_b = mx.symbol.FullyConnected(data = hid_symbol, weight= w_t,
                                        bias=self.v_bias, num_hidden=self.n_hidden)
        p_vh = mx.symbol.Activation(data=wx_b, act_type = 'sigmoid')
        return p_vh

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.binomial_random_visible(v1_mean)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
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
        chain_start = mx.nd.array(chain_start)
      else:
        chain_start = persistent
      for step in xrange(k):
        if step == 0:
          nv_means, nv_samples, \
          nh_means, nh_samples = self.gibbs_hvh(chain_start)
        else:
          nv_means, nv_samples, \
          nh_means, nh_samples = self.gibbs_hvh(nh_samples)

      # w: symbol
      # input: np
      # ph_mean: symbol
      # nv_samples : ndarray
      # nh_means: symbol
      input_symbol = mx.symbol.Variable("input")
      test = input_symbol.bind(mx.cpu(), {"input": mx.nd.array(self.input)})
      nv_samples_symbol = mx.symbol.Variable("nv_samples")
      nv_samples_symbol.bind(mx.cpu(), {"nv_samples": nv_samples})
      self.w = self.w + ( lr * (mx.symbol.dot(mx.symbol.transpose(input_symbol), ph_mean)
                      - mx.symbol.dot(mx.symbol.transpose(nv_samples_symbol), nh_means)) )
      self.v_bias = self.v_bias + (lr * mx.symbol.mean(input_symbol - nv_samples_symbol, axis=0))

      self.h_bias = self.h_bias + (lr * mx.symbol.mean(ph_mean - nh_means, axis=0))

      monitoring_cost = mx.symbol.mean(data=mx.symbol.square(input_symbol - nv_means), axis=0, name="monitoring_cost")
      print(monitoring_cost)
      executor = monitoring_cost.bind(mx.cpu(), {"input": mx.nd.array(self.input)})
      print(executor.forward()[0])
      return monitoring_cost.forward()


    def binomial_random_hidden(self, mean):
      sample_array = mx.nd.array(self.rng.random_sample(size=self.n_hidden))
      return sample_array

    def binomial_random_visible(self, mean):
      sample_array = mx.nd.array(self.rng.random_sample(size=self.n_visible))
      return sample_array



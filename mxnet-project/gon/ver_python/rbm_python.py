
# -*- coding: utf-8 -*-

import numpy
from utils import *

class RBM(object):
  """ Restricted Boltzmann Machine (RBM) """
  def __init__(self, input=None, n_visible=28*28, n_hidden=500, \
      W=None, hbias=None, vbias=None, numpy_rng=None):

    self.input = input
    self.n_visible = n_visible
    self.n_hidden = n_hidden

    if numpy_rng is None:
      numpy_rng = numpy.random.RandomState(1234)

    if W is None:
      W = numpy.asarray(numpy_rng.uniform(
        low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
        high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
        size=(n_visible, n_hidden)))

    if hbias is None:
      hbias = numpy.zeros(n_hidden)

    if vbias is None:
      vbias = numpy.zeros(n_visible)

    self.numpy_rng = numpy_rng
    self.W = W
    self.hbias = hbias
    self.vbias = vbias

  def free_energy(self, v_sample):
    """ no use """
    return

  def propup(self, vis):
    pre_sigmoid_activation = numpy.dot(vis, self.W) + self.hbias
    return sigmoid(pre_sigmoid_activation)

  def sample_h_given_v(self, v0_sample):
    h1_mean = self.propup(v0_sample)
    print(h1_mean)
    h1_sample = self.numpy_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
    return [h1_mean, h1_sample]

  def propdown(self, hid):
    pre_sigmoid_activation = numpy.dot(hid, self.W.T) + self.vbias
    return sigmoid(pre_sigmoid_activation)

  def sample_v_given_h(self, h0_sample):
    v1_mean = self.propdown(h0_sample)
    v1_sample = self.numpy_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean)
    return [v1_mean, v1_sample]

  def gibbs_hvh(self, h0_sample):
    v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
    h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
    return [v1_mean, v1_sample,
            h1_mean, h1_sample]

  def gibbs_vhv(self, v0_sample):
    """ no use """
    return

  def get_cost_updates(self, lr=0.1, persistent=None, k=1):
    ph_mean, ph_sample = self.sample_h_given_v(self.input)

    if persistent is None:
      chain_start = ph_sample
    else:
      chain_start = persistent

    for step in xrange(k):
      if step == 0:
        nv_means, nv_samples,\
        nh_means, nh_samples = self.gibbs_hvh(chain_start)
      else:
        nv_means, nv_samples,\
        nh_means, nh_samples = self.gibbs_hvh(nh_samples)

    self.W += lr * (numpy.dot(self.input.T, ph_mean)
                    - numpy.dot(nv_samples.T, nh_means))
    self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
    self.hbias += lr * numpy.mean(ph_mean - nh_means, axis=0)

    monitoring_cost = numpy.mean(numpy.square(self.input - nv_means))

    return monitoring_cost

  def get_pseudo_likelihood_cost(self, updates):
    """ no use """
    return

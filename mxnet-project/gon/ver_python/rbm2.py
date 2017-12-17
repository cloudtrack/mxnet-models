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
        self.var_data = mx.symbol.Variable('var_data')
        weight = self._weight_generate(self.n_visible, self.n_hidden)
        self.w.bind(ctx=mx.cpu(), args=mx.nd.array(weight))
        self.rng = np.random.RandomState(1234)
        # self.h_samples_ = mx.nd.array(np.zeros((self.batch_size, (n_visible, n_hidden))))
        self.h_samples_ = mx.symbol.Variable('h_samples_')

    def _weight_generate(self, n_visible, n_hidden):
        num_gen = np.random.RandomState(1234)
        w = np.asarray(
                num_gen.uniform(
                    low=-4 * np.sqrt(6. / (n_visible + n_hidden)),
                    high=4 * np.sqrt(6. / (n_visible + n_hidden)),
                    size= (n_visible, n_hidden)
                                ),dtype=np.dtype(np.float16)
                       )
        return w

    def prop_up(self):
        # data = mx.symbol.Variable('data')
        wx_b = mx.symbol.FullyConnected(data = self.var_data, weight = self.w,
                                                    bias = self.h_bias, num_hidden = self.n_hidden)
        activation = mx.symbol.Activation(data=wx_b, act_type = 'sigmoid', name = 'act')
        return activation

    def p_vh(self, hid):
        # hid = self.prop_up()
        w_t = mx.symbol.transpose(data = self.w)
        wx_b = mx.symbol.FullyConnected(data = hid, weight= w_t,
                                        bias=self.v_bias, num_hidden=self.n_hidden)
        p_vh = mx.symbol.Activation(data=wx_b, act_type = 'sigmoid')
        return p_vh


    def p_hv(self, v):
        'v should be symbol'
        wx_b = mx.symbol.FullyConnected(data=v, weight=self.w,
                                        bias=self.h_bias, num_hidden=self.n_hidden)
        p_hv = mx.symbol.Activation(data=wx_b, act_type='sigmoid')
        return p_hv

    def prop_down(self):
        w_t = mx.symbol.transpose(data=self.w)
        wx_b = mx.symbol.FullyConnected(data = self.var_data, weight = w_t,
                                                    bias = self.v_bias, num_hidden = self.n_visible)
        activation = mx.symbol.Activation(data=wx_b, act_type = 'sigmoid', name = 'act')
        return activation

    def _symbol_binomial(self, p):
        sam_arr = mx.nd.array(self.rng.random_sample(size=p.shape))
        sam = mx.sym.Variable('sam')
        sam.bind(ctx=mx.cpu(), args= sam_arr)
        p_sam = sam<sam_arr
        return p_sam

    def _sample_hiddens(self, v):
        """Sample from the distribution P(h|v).
        Parameters"""
        p = self.p_hv(v)
        p_sam = self._symbol_binomial(p)
        return p,p_sam

    def _sample_visable(self, hid):
        """Sample from the distribution P(v|h).
        Parameters"""
        p = self.p_vh(hid)
        p_sam = self._symbol_binomial(p)
        return p,p_sam

    def gibbs_hvh(self, hid):
        v_sam = self._sample_visable(hid)
        h_sam = self._sample_hiddens(v_sam)
        return h_sam

    def gibbs_vhv(self, v):
        h_sam = self._sample_hiddens(v)
        v_sam = self._sample_visable(h_sam)
        return v_sam

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = mx.symbol.FullyConnected(data=v_sample, weight = self.w,
                                        bias=self.h_bias, num_hidden=self.n_hidden)
        vbias_term = mx.symbol.FullyConnected(data=v_sample, weight = self.v_bias,
                                        bo_bias = True, num_hidden=self.n_hidden)
        hidden_term = mx.symbol.sum(data = mx.symbol.log(1+ mx.symbol.exp(data = wx_b)))
        #problem here: parameter of exp should be ndarray

        return -hidden_term - vbias_term

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

        self.W += lr * (numpy.dot(self.input.T, ph_mean)
                        - numpy.dot(nv_samples.T, nh_means))
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * numpy.mean(ph_mean - nh_means, axis=0)

        monitoring_cost = numpy.mean(numpy.square(self.input - nv_means))

        return monitoring_cost

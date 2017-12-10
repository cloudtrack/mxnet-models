# -*- coding: utf-8 -*-

import numpy

numpy.seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))
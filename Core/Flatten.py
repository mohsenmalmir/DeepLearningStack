import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

from pylearn2.sandbox.cuda_convnet.filter_acts   import FilterActs
from theano.sandbox.cuda.basic_ops               import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool          import MaxPool
from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm

# implementing flatten
class Flatten(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a convolutional layer with shared variable internal parameters.
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.tensor4
            
            :type rng: a random number generator used to initialize weights
            """
#        ndim = int(layer_def.find("ndim").text)
#        assert(ndim==2)#for now, only support 1d flattening
        self.init(input,input_shape)
    
    """Pool Layer of a convolutional network """
    def init(self, input, input_shape):
        """
            
            :type input: theano.tensor.dtensor4
            :arrangement: c01b
            
            :type ndim: input to the flatten
            :param
            """
        self.input         = input
        batch_size         = input_shape[-1]#c01b
        self.output        = T.reshape(input,[np.prod(input_shape[:-1]),batch_size])
        self.input_shape   = input_shape
        self.output_shape  = [np.prod(input_shape[:-1]),batch_size]
        self.params        = []













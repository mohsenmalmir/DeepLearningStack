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
            Create a flatten layer, which converts a theano.tensor4 to theano.matrix by flattening across width and height, e.g. dimensions 1 and 2 
            
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
            
            :type input_shape: tuple or list containing the size of the input 
            """
        self.input         = input
        batch_size         = input_shape[-1]#c01b
        self.output        = T.reshape(input,[np.prod(input_shape[:-1]).astype(np.int32),batch_size])
        self.input_shape   = input_shape
        self.output_shape  = [np.prod(input_shape[:-1]).astype(np.int32),batch_size]
        self.params        = []













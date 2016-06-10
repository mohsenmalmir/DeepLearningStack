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

# implementing local response normalization
class LRN(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a local response normalization layer, using crossmap normalization within the specified neighborhood size 
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.tensor4
            
            :type rng: a random number generator used to initialize weights
            """
        normSizeAlphaBeta = [ int(layer_def.find("normsize").text),
                              float(layer_def.find("normalpha").text),
                              float(layer_def.find("normbeta").text)]
        
        self.init(input, input_shape, normSizeAlphaBeta)
    
    """Pool Layer of a convolutional network """
    def init(self, input, input_shape, normSizeAlphaBeta):
        """
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.
            
            
            type normSizeAlphaBeta: tuple or list of size 3
            :param normSizeAlphaBeta: (size,alpha,beta)
            """

        self.input         = gpu_contiguous(input)
        #NORM
        norm_op            = CrossMapNorm( size_f=normSizeAlphaBeta[0], add_scale=normSizeAlphaBeta[1], pow_scale=normSizeAlphaBeta[2], blocked=True)
        self.output        = norm_op(self.input)[0]
        
        self.input_shape   = input_shape
        self.output_shape  = input_shape
    
        self.params        = []













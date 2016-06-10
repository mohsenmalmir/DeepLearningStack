import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing flatten
class Concatenate(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create a concatenation layer
            The inputs is a list of matrices, the output is the concatenation of these matrices across dimension 0
            
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: list or tuple of variables to concatenate
            
            :type inputs_shape: list or tuple of the size of inputs
            
            :type rng: a random number generator used to initialize weights
            """
        self.inputs        = inputs
        output_dim         = np.sum([sz[0] for sz in inputs_shape])
        batch_size         = inputs_shape[0][1]
        self.output        = T.concatenate(inputs,axis=0)
        self.input_shape   = inputs_shape
        self.output_shape  = [output_dim,batch_size]
        self.params        = []













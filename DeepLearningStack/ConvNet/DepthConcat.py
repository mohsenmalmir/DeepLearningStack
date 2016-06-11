import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing flatten
class DepthConcat(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create a depth concatenation layer. This is for concatenating outputs of convolutional layers, for example to create inception modules 
            The output is created by concatenating the inputs along dimension 0
            
            :type layer_def: Element, xml containing configu for Conv layer

            :type inputs: a list of theano.tensor4, all should have the same height and width
            
            :type input_shapes: list of sizes of each tensor4 

            :type rng: a random number generator used to initialize weights
        """
        self.inputs        = inputs
        n_channels         = np.sum([sz[0] for sz in inputs_shape])
        batch_size         = inputs_shape[0][3]
        dim1               = inputs_shape[0][1]
        dim2               = inputs_shape[0][2]
        for sz in inputs_shape:
            assert(dim1       == sz[1])
            assert(dim2       == sz[2])
            assert(batch_size == sz[3])
        self.output        = T.concatenate(inputs,axis=0)
        self.input_shape   = inputs_shape
        self.output_shape  = [n_channels,dim1,dim2,batch_size]
        self.params        = []













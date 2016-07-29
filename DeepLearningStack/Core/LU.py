from __future__ import print_function
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous


# implementing linear unit
class LU(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a linear unit, which simply multiplies the input to a weight matrix to produce the output 
            output is simply input * W + b, where b is the bias
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.matrix

            :type input_shape: list or tuple of the size of the input 
            
            :type rs: a random state
            """
        layer_name= layer_def.attrib["name"]
        numUnits  = int(layer_def.find("numunits").text)
        init_bias = float(layer_def.find("bias").text)
        
        assert(len(input_shape)==2)
        if clone_from!=None:
            clone_coeff = float(layer_def.find("CloneCoeff").text)

        rng             = np.random.RandomState(seed=int(time.time()))
        self.input      = input
        n_in,batch_size = input_shape
        # initialize weights with random weights
        if clone_from!=None:
            self.W   = clone_from.W
            self.b   = clone_from.b
        else:
            W_bound  = 0.01#numpy.sqrt(6. / (n_in + n_out))
            W_values = np.asarray(rng.normal(loc=0., scale=W_bound, size=(numUnits, n_in)), dtype=theano.config.floatX)
            self.W   = theano.shared(value=W_values, name=layer_name+"-W", borrow=True)
            b_values = init_bias * np.ones((numUnits,), dtype=theano.config.floatX)
            self.b   = theano.shared(value=b_values, name=layer_name+"-b", borrow=True)


#        self.output       = T.dot(gpu_contiguous(self.W), input) + T.tile(b.reshape([numUnits,1]),[1,batch_size])#gpu_contiguous(T.tile(self.b,[1,batch_size]))
        if clone_from!=None:
            self.output       = T.dot(clone_coeff * self.W, input) + clone_coeff * self.b.dimshuffle(0, 'x')
        else:
            self.output       = T.dot(self.W, input) + self.b.dimshuffle(0, 'x')
        # parameters of the model
        self.input_shape  = input_shape
        self.output_shape = [numUnits,batch_size]
        self.params       = [self.W,self.b]


import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous


# implementing linear unit
class LSTM(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create an LSTM layer with shared variable internal parameters.
            This is a fully connected layer, with inputs from the previous step,
            We used a forget gate only, output = (1-forget) * prev_out + forget * input
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type inputs: list of inputs [input,gate_input,prev_output] 
            :param inputs[0]: input, the input which is a theano.matrix
            :param inputs[1]: gate_input, the input based on which the gate activities are determined
            :param inputs[2]: prev_output, the feedback input to LSTM
            
            :type rs: a random state
            """

        #inputs = [input,gate_input,previous_output]
        assert(len(inputs)==3)
        assert(len(inputs_shape)==3)
        self.input      = inputs[0]
        self.gate_input = inputs[1]
        self.prev_out   = inputs[2]
        dim,batch_size  = inputs_shape[0]
        gatein_dim,bsz  = inputs_shape[1]
        assert(bsz == batch_size)
        
        # clone the num_units 
        if clone_from==None:
            self.num_units = int(layer_def.find("numunits").text)
        else:
            self.num_units = clone_from.num_units

        assert(inputs_shape[2][0] == self.num_units)
        assert(inputs_shape[2][1] == batch_size)
        init_bias = float(layer_def.find("bias").text)

        rng             = np.random.RandomState(seed=int(time.time()))
        # initialize weights with random weights
        if clone_from!=None:
            self.W   = clone_from.W
            self.b   = clone_from.b
        else:
            W_bound  = np.sqrt(6. / (dim + gatein_dim))
            W_values = np.asarray(rng.normal(loc=0., scale=W_bound, size=(dim, gatein_dim)), dtype=theano.config.floatX)
            self.W   = theano.shared(value=W_values, name='W', borrow=True)
            b_values = init_bias * np.ones((dim,), dtype=theano.config.floatX)
            self.b   = theano.shared(value=b_values, name='b', borrow=True)


        self.forget       = T.nnet.sigmoid( T.dot(self.W, self.gate_input) + self.b.dimshuffle(0, 'x') )
        self.output       = ( 1. - self.forget ) * self.prev_out + self.forget * self.input 
        # parameters of the model
        self.inputs_shape = inputs_shape
        self.output_shape = [self.num_units,batch_size]
        self.params       = [self.W,self.b]


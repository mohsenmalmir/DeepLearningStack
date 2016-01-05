import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing naive bayes belief update: P(O | b1,b2)  \propto P(O|b1) * P(O|b2)
class NaiveBayesBeliefUpdate(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create a concatenation layer
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: list or tuple of variables to concatenate
            
            :type input_shape: list or tuple of the size of inputs
            
            :type rng: a random number generator used to initialize weights
            """
        self.inputs                  = inputs
        if len(inputs)==2:
            assert(inputs_shape[0][0] == inputs_shape[1][0])
            output_dim                   = inputs_shape[0][0]
            batch_size                   = inputs_shape[0][1]
            temp                         = inputs[0] * inputs[1]
        elif len(inputs)==3:
            assert(inputs_shape[0][0] == inputs_shape[1][0])#data size
            assert(inputs_shape[0][1] == inputs_shape[1][1])#batch size
            assert(inputs_shape[0][1] == inputs_shape[2][1])#batch size
            assert(inputs_shape[2][0] == 1)
            output_dim                   = inputs_shape[0][0]
            batch_size                   = inputs_shape[0][1]
            temp                         = (inputs[0]**inputs[2]) * inputs[1]
        self.output                  = temp / T.reshape(T.sum(temp,axis=0),[1,batch_size])
        self.inputs_shape            = inputs_shape
        self.output_shape            = [output_dim,batch_size]
        self.params                  = []













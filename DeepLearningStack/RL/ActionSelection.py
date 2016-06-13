from __future__ import print_function
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing softmax operation
class ActionSelection(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create an action selection layer for RL deepnet
            The action selection either picks the 'max' value action, or 'soft' action with probability proportional to its value
            
            :type layer_def: Element, xml containing configuration for this layer
            
            :type input: tensor.matrix, 2D
            
            :type input_shape: tuple or list of size 2
            :param input_shape: [c,b] b is the batch_size, c is data dimensionality
            
            :type rs: a random number generator used to initialize weights
            """
        mechanism         = layer_def.find("type").text
        self.input        = input
        self.input_shape  = input_shape
        num_acts,batch_sz = input_shape
        self.output_shape = [1,batch_sz]#a vector of size b
        if mechanism=="max":#select max action
            actions       = T.argmax(self.input,axis=0)
        elif mechanism=="soft":#the input should be a softmax, then this selects an action randomly
            temp1         = input - T.min(input)
            temp2         = temp1 / temp1.sum(axis=0).reshape([1,batch_sz])
            self.cumsum   = T.extra_ops.cumsum(temp2,axis=0)
            self.rndunif  = rs.uniform(size=(1,input_shape[1]), low=0.0, high=1.0)
            selectedActs  = T.ge(self.cumsum,self.rndunif).astype(theano.config.floatX)
            actions       = T.argmax(selectedActs,axis=0)
        else:
            print ("invalid action-selection type")
            raise
        self.output       = actions.reshape([1,batch_sz])
        self.params       = []


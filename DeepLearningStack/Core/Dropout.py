from __future__ import print_function
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing rectification operation
class Dropout(object):
    
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a dropout layer, usually used after a fully connected layer for regularization 
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.matrix
            
            :type input_shape: tuple or list of size 2
            :param input_shape: [channels,batchsize] c01b
            
            :type rs: a random number generator used to initialize weights

            :type clone_from: indicates that this is a clone, therefore the dropout is replaced by averaging 
                               CloneP is the probability used to multiply to the weights
                               CloneP should be the same as P, but is left separate for flexibility
            """
        P = float(layer_def.find("Prob").text)
        if clone_from!=None:
            P = float(layer_def.find("CloneP").text)
        self.init(input,input_shape, P,rs)

    def init(self,input, input_shape, P,rs):
        self.input        = input
        self.drop         = rs.binomial(input_shape,p=P).astype(theano.config.floatX)
        self.output       = self.drop * input
        self.input_shape  = input_shape
        self.output_shape = input_shape
        self.params       = []













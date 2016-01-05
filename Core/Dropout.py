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
            Create a convolutional layer with shared variable internal parameters.
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.tensor4
            
            :type input_shape: tuple or list of size 4
            :param input_shape: [channels,height,width,batchsize] c01b
            
            :type rs: a random number generator used to initialize weights
            """
        P = float(layer_def.find("Prob").text)
        if clone_from!=None:
            P = float(layer_def.find("CloneP").text)
        self.init(input,input_shape, P,rs)

    def init(self,input, input_shape, P,rs):
        self.input        = input
        self.drop         = rs.binomial(input_shape,p=P).astype(theano.config.floatX)
        print type(self.drop)
        self.output       = self.drop * input
        self.input_shape  = input_shape
        self.output_shape = input_shape
        self.params       = []













import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing softmax operation
class Tanh(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a Tanh layer 
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.matrix
            
            :type input_shape: tuple or list of size 4
            :param input_shape: [channels,height,width,batchsize] c01b
            
            :type rs: a random number generator used to initialize weights
            """
        self.init(input, input_shape)
    
    def init(self, input, input_shape):
        self.input        = input
        self.output       = T.tanh(input) #tanh IS ELEMENT-WISE
        self.input_shape  = input_shape
        self.output_shape = input_shape
        self.params = []


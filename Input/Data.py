import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing Rectified linear unit
class Data(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a input data layer that is 2D, with features lying in rows.
            
            :type layer_def: Element, xml containing configu for Conv layer

            :type input: theano.matrix
            
            :type input_shape: ignored for this class

            :type rng: a random number generator used to initialize weights
        """
        batch_size      = int(layer_def.find("batchsize").text)
        data_size       = int(layer_def.find("datasize").text)
        self.layer_name = layer_def.attrib["name"]
        self.init(input, input_shape, data_size, batch_size)#input is a matrix of size data_size x batch_size
    
    
    """Pool Layer of a convolutional network """
    def init(self, input, input_shape, data_size, batch_size):
        """
            Create an image layer for deep net.
            
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights
            
            :type input: None

            :type batch_size: image batch size

            :type image_size: dimensions of the image
            
            :type image channels: number of image channels


            """
        

        self.input_shape   = input_shape
        self.output_shape  = [data_size, batch_size]
        self.input         = input
        self.output        = input
        self.params        = []













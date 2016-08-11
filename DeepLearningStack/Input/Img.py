import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing Rectified linear unit
class Img(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a 4D input to the network, theano.tensor4, with arrangement of the input as c01b 
            
            :type layer_def: Element, xml containing configu for Conv layer

            :type input: tensor.tensor4
            
            :type input_shape: ignored for this class

            :type rng: a random number generator used to initialize weights
        """
        batch_size      = int(layer_def.find("batchsize").text)
        image_size      = int(layer_def.find("imagesize").text)
        image_channels  = int(layer_def.find("imagechannels").text)
        self.layer_name = layer_def.attrib["name"]
        self.init(input, input_shape, batch_size,image_size,image_channels)
    
    
    def init(self, input, input_shape, batch_size, image_size, image_channels):
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
        _,size0,size1,_    = input_shape
        self.output_shape  = [image_channels,size0,size1,batch_size]#c01b
        self.input         = input
        self.output        = input
        self.params        = []













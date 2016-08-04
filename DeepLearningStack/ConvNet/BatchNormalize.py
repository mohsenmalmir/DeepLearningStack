import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing softmax operation
class BatchNormalize(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a batch normalization layer, from the following paper:
            Ioffe S, Szegedy C. Batch normalization: Accelerating deep 
                    network training by reducing internal covariate shift. 
                    arXiv preprint arXiv:1502.03167. 2015 Feb 11. 
            Each input channel is normalized to 0 mean and 1 std 
            Given an input x, the output y is defined as:
                y       = gamma * xhat + beta
                xhat    = (x - mean) / (std + epsilon)
                where mean and std are calculated across batch and channel (dimension 0 of the input)
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.tensor4
            
            :type input_shape: tuple or list of size 4
            :param input_shape: [channels,height,width,batchsize] c01b
            
            :type rs: a random number generator used to initialize weights
            """
        self.epsilon      = float(layer_def.find("epsilon").text)
        layer_name        = layer_def.attrib["name"]
    
        self.input        = input




        if len(input_shape)==4:
            nc,h,w,b          = input_shape
            if clone_from==None: 
                self.gamma    = theano.shared( np.ones(nc).astype(theano.config.floatX) ,borrow=True,name=layer_name+"-gamma")
                self.beta     = theano.shared( np.zeros(nc).astype(theano.config.floatX),borrow=True,name=layer_name+"-beta")
            else:
                self.gamma    = clone_from.gamma 
                self.beta     = clone_from.beta 
            #the variance is calculated over batches and over a feature map
            std               = input.std(axis=[0,3],keepdims=True) + self.epsilon#take the std over batches
            mn                = input.mean(axis=[0,3],keepdims=True)
            self.output       = self.gamma.dimshuffle(0, 'x', 'x','x') * ( (input - mn)/ std ) + self.beta.dimshuffle(0, 'x', 'x','x') #normalize each dimension
        elif len(input_shape)==2:
            n,b               = input_shape
            if clone_from==None: 
                self.gamma    = theano.shared( np.ones(n).astype(theano.config.floatX) ,borrow=True,name=layer_name+"-gamma")
                self.beta     = theano.shared( np.zeros(n).astype(theano.config.floatX),borrow=True,name=layer_name+"-beta")
            else:
                self.gamma    = clone_from.gamma 
                self.beta     = clone_from.beta 
            #the variance is calculated over batches and over a feature map
            std               = input.std(axis=1,keepdims=True) + self.epsilon#take the std over batches
            mn                = input.mean(axis=1,keepdims=True)
            self.output       = self.gamma.dimshuffle(0, 'x') * ( (input - mn)/ std ) + self.beta.dimshuffle(0, 'x') #normalize each dimension
        self.input_shape  = input_shape
        self.output_shape = input_shape
        self.params       = [self.gamma,self.beta]


import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous


# implementing rectification operation
class Rectifier(object):
    
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a rectification layer 
            The rectification is a1 * (x>0)+ a2 * (x<0), with a1 and a2 being learned or constant
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.matrix
            
            :type input_shape: tuple or list of size 4
            :param input_shape: [channels,height,width,batchsize] c01b
            
            :type rs: a random number generator used to initialize weights
            """
        posslope = [ float(layer_def.find("posslope").find("min").text),
                     float(layer_def.find("posslope").find("max").text),
                     float(layer_def.find("posslope").find("init").text),
                     layer_def.find("posslope").find("method").text]

        negslope = [ float(layer_def.find("negslope").find("min").text),
                    float(layer_def.find("negslope").find("max").text),
                    float(layer_def.find("negslope").find("init").text),
                    layer_def.find("negslope").find("method").text]
        self.init(input,input_shape, posslope, negslope)

    def init(self,input, input_shape, posslope, negslope):
        self.input  = input
#        assert( posslope[-1]=="Constant")#learning slope not yet implemented
#        assert( negslope[-1]=="Constant")
        #
        if negslope[-1]=="Constant":
            self.output       = T.ge(input , 0.) * input * posslope[2] + T.lt(input, 0.) * input * negslope[2]
            self.input_shape  = input_shape
            self.output_shape = input_shape
            self.params       = []
        elif negslope[-1]=="Learn":
            self.pos_slope_sym     = theano.shared(posslope[2]*np.ones([input_shape[0],1],dtype=theano.config.floatX),borrow=True,broadcastable=(False,True))
            self.neg_slope_sym     = theano.shared(negslope[2]*np.ones([input_shape[0],1],dtype=theano.config.floatX),borrow=True,broadcastable=(False,True))
            self.output            = T.ge(input,0) * input * T.ge(self.pos_slope_sym , posslope[0]) * T.le(self.pos_slope_sym , posslope[1]) * self.pos_slope_sym + T.lt(input,0) * input * T.ge(self.neg_slope_sym , negslope[0]) * T.le(self.neg_slope_sym , negslope[1]) * self.neg_slope_sym
            self.params            = [self.pos_slope_sym,self.neg_slope_sym]
            self.input_shape       = input_shape
            self.output_shape      = input_shape
        else:
            assert(False)













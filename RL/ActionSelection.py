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
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.matrix
            
            :type input_shape: tuple or list of size 2
            :param input_shape: [c,b] b is the batch_size, c is data dimensionality
            
            :type rs: a random number generator used to initialize weights
            """
        mechanism         = layer_def.find("type").text
#        epsilon           = float(layer_def.find("epsilon").text)
#        assert(epsilon>=0 and epsilon<=1.)
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
            print "invalid action-selection type"
            raise
        #epsilon-greedy
#        if epsilon>0.:
#            #select with probability epsilon to do a random action for each data in the batch
#            rndaction     = rs.binomial((1,batch_sz),p=epsilon).astype(theano.config.floatX).reshape([-1])
#            rndidx        = rs.random_integers((1,batch_sz),low=0,high=num_acts-1).reshape([-1])#randomly selected action
#            actions       = rndaction * rndidx + (1.-rndaction) * actions#selected actions for each game in the batch
        #the output of each layer should be at least 2 dimensions
        self.output       = actions.reshape([1,batch_sz])
#        self.cumsum       = rndunif
        self.params       = []


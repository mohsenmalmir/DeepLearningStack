import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous


# implementing Gaussian observation layer
# each output unit encodes P(o | s,a) ~ N(o | mu(s,a) , sigma(s,a) ) 
class GaussianObs(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        #the inputs are expected to be [actions,features]
        numLabels   = int(layer_def.find("numlabels").text)
        numActions  = int(layer_def.find("numactions").text)
        featdim     = int(layer_def.find("featdim").text)
        covType     = layer_def.find("covariance").text 
        assert(covType == "diag")
        actions,feats = inputs
        #assert(feats.output_shape[0] == featdim)

        #initialize weights
        rng             = np.random.RandomState(seed=int(time.time()))
        self.inputs     = inputs
        _,batch_size    = inputs_shape[0]
        #the parameters of this model are a matrix of size mu = n_in x numUnits, var= n_in x numUnits 
        # initialize mean and variance 
        if clone_from!=None:
            self.mu     = clone_from.mu
            self.std    = clone_from.std
        else:
            values   = np.ones([numActions ,numLabels ,featdim ],dtype=theano.config.floatX) 
            self.mu  = theano.shared(value=values, name='mu', borrow=True)
            self.std = theano.shared(value=values, name='std', borrow=True)

        #calculate output
        #input : featdim x batchsize
        #mu: numActions x numLabels x featdim
        #std: numActions x numLabels x featdim
                        #batchsize x numLabels x featdim 
        mean_subtracted     = (self.mu[actions,:,:] - feats.dimshuffle(1,'x',0))/ self.std[actions,:,:]# output: batchsize x numLabels x featdim 
        Xsq                 = -0.5 * (mean_subtracted * mean_subtracted).sum(axis=2)
        temp                = -0.5*featdim*np.log(2.*np.pi)-T.sum(T.log(self.std[actions,:,:]),axis=2).dimshuffle(1,0)+Xsq.dimshuffle(1,0) 

        """
        mean_subbed = (means[actions_np,:,:] - x.T[:,np.newaxis,:]) /  stds[actions_np,:,:]
        Xsq = -0.5 * (mean_subbed*mean_subbed).sum(axis=2)
        coeff = -0.5 * 16384 * np.log(2 * np.pi) - np.sum(np.log(stds[actions_np,:,:]),axis=2).T + Xsq.T
        """

        #self.output         = T.exp(temp / 1000.)
        self.output         = temp
        # parameters of the model
        self.inputs_shape   = inputs_shape
        self.output_shape   = [numLabels,batch_size]
        self.params         = [self.mu,self.std]
        #self.params         = []


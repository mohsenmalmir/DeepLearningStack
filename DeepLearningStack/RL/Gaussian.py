import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous


# implementing Gaussian layer
# this is for now limited to diagonal covariance matrix
class Gaussian(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a convolutional layer with shared variable internal parameters.
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.matrix
            
            :type rs: a random state
            """
        numUnits  = int(layer_def.find("numunits").text)
        covType   = layer_def.find("covariance").text 
        assert(covType == "diag")

        
        assert(len(input_shape)==2)

        rng             = np.random.RandomState(seed=int(time.time()))
        self.input      = input
        n_in,batch_size = input_shape
        assert(n_in == numUnits)
        #the parameters of this model are a matrix of size mu = n_in x numUnits, var= n_in x numUnits 
        # initialize mean and variance 
        if clone_from!=None:
            self.mu     = clone_from.mu
            self.cov    = clone_from.cov
        else:
            W_bound  = .1#numpy.sqrt(6. / (n_in + n_out))
            W_values = np.asarray(rng.normal(loc=0., scale=W_bound, size=(n_in, numUnits)), dtype=theano.config.floatX)
            self.mu  = theano.shared(value=W_values, name='mu', borrow=True)
            #for now only consider fixed variances=1
            b_values = np.ones([n_in, numUnits], dtype=theano.config.floatX)
            #self.cov = theano.shared(value=np.abs(b_values)+.2, name='cov', borrow=True)
            self.cov = theano.shared(value=b_values, name='cov', borrow=True)
            #print self.mu.get_value().min(),self.mu.get_value().max(),self.cov.get_value().min(),self.cov.get_value().max()
            #exit(0)


        #calculating the output


        mean_subtracted     = self.mu.dimshuffle('x',0,1) - input.dimshuffle(1,0,'x')# output: batchsize x n_in x numUnits  
        #since unit variance
        divide_by_cov       = (mean_subtracted*mean_subtracted) #/ self.cov.dimshuffle('x',0,1)
        #sum over the data dimension, which is dimension 1
        # the computations are done in log space
        #exp_term            = T.exp(-0.5*divide_by_cov.sum(axis=1).dimshuffle(1,0))# the result should be numUnits x batch_size 
        #self.output         = (1./numUnits) * (1. / ( np.sqrt(2.*np.pi) * T.sqrt(T.prod(self.cov,axis=0)) )).dimshuffle(0,'x') * exp_term 
        
        exp_term            = -0.5*divide_by_cov.sum(axis=1).dimshuffle(1,0)# the result should be numUnits x batch_size 
        #temp                = -0.5*n_in*np.log(2.*np.pi)-0.5*T.sum(T.log(self.cov),axis=0).dimshuffle(0,'x')+exp_term 
        temp                = -0.5*n_in*np.log(2.*np.pi)+exp_term 
        self.output         = (1./numUnits) * T.exp(temp)
        #self.output         = temp2 / T.sum(temp2,axis=0).reshape([1,-1])
        #self.output         = T.exp(temp)
        #self.output         = temp
        #self.output         = temp#output is the log-prob
        # parameters of the model
        self.input_shape    = input_shape
        self.output_shape   = [numUnits,batch_size]
        self.params         = [self.mu]
        #self.params         = []


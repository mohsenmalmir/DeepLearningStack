import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous

"""
The implementation in this code follows the details of:

Cho K, Van Merrienboer B, Bahdanau D, Bengio Y. 
On the properties of neural machine translation: 
Encoder-decoder approaches. 
arXiv preprint arXiv:1409.1259. 2014 Sep 3.
"""

# implementing linear unit
class GRU(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create an Gated Recurrent Unit layer with shared variable internal parameters.
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type inputs: list of inputs [input,gate_input,prev_output] 
            :param inputs[0]: input, the input which is a theano.matrix, x_t
            :param inputs[1]: previous state, h_{t-1}, same shape as this layer
            
            :type rs: a random state
            """

        #inputs = [input,gate_input,previous_output]
        layer_name= layer_def.attrib["name"]
        assert(len(inputs)==2)
        assert(len(inputs_shape)==2)
        self.input      = inputs[0]
        self.prev_h     = inputs[1]
        n_in,_          = inputs_shape[0]
        n_prev_h,bsz    = inputs_shape[1]
        assert(bsz == inputs_shape[0][1])
        
        # clone the num_units 
        if clone_from==None:
            self.num_units = int(layer_def.find("numunits").text)
        else:
            self.num_units = clone_from.num_units

        assert(n_prev_h == self.num_units)

        #create the weight matrices
        rng             = np.random.RandomState(seed=int(time.time()))
        # initialize weights with random weights
        if clone_from!=None:
            #weight matrices for x_t, the input
            self.W_z    = clone_from.W_z
            self.W_r    = clone_from.W_r
            self.W      = clone_from.W
            #weight matrices for h_{t-1}
            self.U_z    = clone_from.U_z
            self.U_r    = clone_from.U_r
            self.U      = clone_from.U
        else:
            #W_{}: is a matrix of size num_units x n_in
            W_bound     = np.sqrt(6. / (self.num_units + n_in))
            #W_o
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W_z    = theano.shared(value=W_values, name=layer_name+'-Wz', borrow=False)# num_units x n_in
            #W_f
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W_r    = theano.shared(value=W_values, name=layer_name+'-Wr', borrow=False)# num_units x n_in
            #W_i
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W      = theano.shared(value=W_values, name=layer_name+'-W', borrow=False)# num_units x n_in

            #U_{}: is a matrix of size num_units x num_units 
            U_bound     = np.sqrt(6. / (self.num_units +self.num_units))
            #U_o
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U_z    = theano.shared(value=U_values, name=layer_name+'-Uz', borrow=False)#num_units x num_units
            #U_f
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U_r    = theano.shared(value=U_values, name=layer_name+'-Ur', borrow=False)#num_units x num_units
            #U_i
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U      = theano.shared(value=U_values, name=layer_name+'-U', borrow=False)#num_units x num_units

        #calculate the gate values
                                            # num_units x bsz               #num_units x bsz
        self.zgate      = T.nnet.sigmoid( T.dot(self.W_z, self.input) + T.dot(self.U_z,self.prev_h) )#update gate
                                            # num_units x bsz               #num_units x bsz
        self.rgate      = T.nnet.sigmoid( T.dot(self.W_r, self.input) + T.dot(self.U_r,self.prev_h) )#reset gate
                                            # num_units x bsz       #num_units x bsz
        self.tilde_h    = T.tanh( T.dot(self.W, self.input) + T.dot(self.U,( self.rgate * self.prev_h) ) )#new memory content
        #output is a dictionary
        #only if there is a mem output tag, then provide this output
        self.output     = dict()
        self.output_shape= dict()
        #the default output
        self.output[layer_name]         = (1. - self.zgate) * self.prev_h + self.zgate * self.tilde_h  
        self.output_shape[layer_name]   = [self.num_units,bsz]
        self.inputs_shape               = inputs_shape
        # parameters of the model
        if clone_from==None:
            self.params   = [self.W_z,self.W_r,self.W,self.U_z, self.U_r, self.U]
        else:
            self.params   = []


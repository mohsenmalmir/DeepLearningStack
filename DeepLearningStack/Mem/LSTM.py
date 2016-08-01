import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous

"""
The implementation in this code follows the details of:

Graves A. Generating sequences with recurrent neural networks. 
arXiv preprint arXiv:1308.0850. 2013 Aug 4.
"""

# implementing linear unit
class LSTM(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create an LSTM layer with shared variable internal parameters.
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type inputs: list of inputs [input,gate_input,prev_output] 
            :param inputs[0]: input, the input which is a theano.matrix, x_t
            :param inputs[1]: previous state, h_{t-1}, same shape as this layer
            :param inputs[2]: previous memory, c_{t-1}, same shape as this layer
            
            :type rs: a random state
            """

        #inputs = [input,gate_input,previous_output]
        layer_name= layer_def.attrib["name"]
        assert(len(inputs)==3)
        assert(len(inputs_shape)==3)
        self.input      = inputs[0]
        self.prev_h     = inputs[1]
        self.prev_c     = inputs[2]
        n_in,_          = inputs_shape[0]
        n_prev_v,bsz    = inputs_shape[1]
        assert(bsz == inputs_shape[0][1])
        assert(bsz == inputs_shape[2][1])
        #make sure previous h,c have the same dimensions
        assert(inputs_shape[2][0] == inputs_shape[1][0])
        assert(inputs_shape[2][1] == inputs_shape[1][1])
        
        # clone the num_units 
        if clone_from==None:
            self.num_units = int(layer_def.find("numunits").text)
        else:
            self.num_units = clone_from.num_units

        assert(n_prev_v == self.num_units)

        #create the weight matrices
        rng             = np.random.RandomState(seed=int(time.time()))
        # initialize weights with random weights
        if clone_from!=None:
            #weight matrices for x_t, the input
            self.W_o    = clone_from.W_o
            self.W_f    = clone_from.W_f
            self.W_i    = clone_from.W_i
            self.W_c    = clone_from.W_c
            #weight matrices for h_{t-1}
            self.U_o    = clone_from.U_o
            self.U_f    = clone_from.U_f
            self.U_i    = clone_from.U_i
            self.U_c    = clone_from.U_c
            #weight matrices (diagonal) for previous memory c_{t-1}
            self.V_o    = clone_from.V_o
            self.V_f    = clone_from.V_f
            self.V_i    = clone_from.V_i
        else:
            #W_{}: is a matrix of size num_units x n_in
            W_bound     = np.sqrt(6. / (self.num_units + n_in))
            #W_o
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W_o    = theano.shared(value=W_values, name=layer_name+'-Wo', borrow=False)
            #W_f
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W_f    = theano.shared(value=W_values, name=layer_name+'-Wf', borrow=False)
            #W_i
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W_i    = theano.shared(value=W_values, name=layer_name+'-Wi', borrow=False)
            #W_c
            W_values    = np.asarray(rng.normal(loc=0., scale=W_bound, size=(self.num_units, n_in)), dtype=theano.config.floatX)
            self.W_c    = theano.shared(value=W_values, name=layer_name+'-Wc', borrow=False)

            #U_{}: is a matrix of size num_units x num_units 
            U_bound     = np.sqrt(6. / (self.num_units +self.num_units))
            #U_o
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U_o    = theano.shared(value=U_values, name=layer_name+'-Uo', borrow=False)
            #U_f
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U_f    = theano.shared(value=U_values, name=layer_name+'-Uf', borrow=False)
            #U_i
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U_i    = theano.shared(value=U_values, name=layer_name+'-Ui', borrow=False)
            #U_c
            U_values    = np.asarray(rng.normal(loc=0., scale=U_bound, size=(self.num_units, self.num_units)), dtype=theano.config.floatX)
            self.U_c    = theano.shared(value=U_values, name=layer_name+'-Uc', borrow=False)

            #V_{}: is a diagonal matrix of size num_units x num_units 
            V_bound     = np.sqrt(6. / (self.num_units+self.num_units ))
            #U_o
            V_values    = np.asarray(rng.normal(loc=0., scale=V_bound, size=(self.num_units, )), dtype=theano.config.floatX)
            self.V_o    = theano.shared(value=V_values, name=layer_name+'-Vo', borrow=False)
            #U_f
            V_values    = np.asarray(rng.normal(loc=0., scale=V_bound, size=(self.num_units, )), dtype=theano.config.floatX)
            self.V_f    = theano.shared(value=V_values, name=layer_name+'-Vf', borrow=False)
            #U_i
            V_values    = np.asarray(rng.normal(loc=0., scale=V_bound, size=(self.num_units, )), dtype=theano.config.floatX)
            self.V_i    = theano.shared(value=V_values, name=layer_name+'-Vi', borrow=False)


        #calculate the gate values
        self.fgate      = T.nnet.sigmoid( T.dot(self.W_f, self.input) + T.dot(self.U_f,self.prev_h) + self.V_f.dimshuffle(0,'x') * self.prev_c )#forget gate
        self.igate      = T.nnet.sigmoid( T.dot(self.W_i, self.input) + T.dot(self.U_i,self.prev_h) + self.V_i.dimshuffle(0,'x') * self.prev_c )#input gate
        self.tilde_c    = T.tanh(    T.dot(self.W_c, self.input) + T.dot(self.U_c,self.prev_h) )#new memory content
        self.c          = self.fgate * self.prev_c + self.igate * self.tilde_c#updated memory content
        self.ogate      = T.nnet.sigmoid( T.dot(self.W_o, self.input) + T.dot(self.U_o,self.prev_h) + self.V_o.dimshuffle(0,'x') * self.c )#output gate
        #output is a dictionary
        #only if there is a mem output tag, then provide this output
        self.output     = dict()
        self.output_shape= dict()
        multiouts       = layer_def.findall("output")
        for out in multiouts:
            if out.attrib["type"].lower()=="mem":
                self.output[out.text] = self.c
                self.output_shape[out.text] = [self.num_units,bsz]
        #the default output
        self.output[layer_name] = self.ogate * T.tanh(self.c)
        self.output_shape[layer_name] = [self.num_units,bsz]
        self.inputs_shape = inputs_shape
        # parameters of the model
        if clone_from==None:
            self.params   = [self.W_o,self.W_f,self.W_i,self.W_c,self.U_o, self.U_f, self.U_i, self.U_c, self.V_o, self.V_f,self.V_i]
        else:
            self.params   = []


import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T



# implementing softmax operation
class DirichletLayer(object):
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create a Dirichlet layer, according to the following paper:
                Malmir M, Sikka K, Forster D, Fasel I, Movellan JR, Cottrell GW. 
                Deep Active Object Recognition by Joint Label and Action Prediction. 
                arXiv preprint arXiv:1512.05484. 2015 Dec 17. 
            Each unit in this layer encodes a Dicihlet distribution over its input.
            The input is assumed to be a belief vector, i.e. \sum_i input[i] = 1, 0 <= input_i <= 1 for all i
                 
            :type layer_def: Element, xml containing configu for Conv layer

            :type inputs: a list of [belief_in, actions, objects, previous_belief] 
            :param inputs[0], belief_in, is a theano.matrix which contains belief vectors in its columns
            :param inputs[1], actions, theano.ivector, list of actions for each column of belief_in 
            :param inputs[2], objects, theano.ivector, list of objects for each column of belief_in 
            :param inputs[3], previous_belief, theano.matrix, used to accumulate beliefs over time 
            
            
            :type input_shapes: list of sizes of inputs 

            :type rs: a random number generator used to initialize weights
        """
        assert(len(inputs) == 4)#belief dim x bacth_sz, actions: 1 x batch_size, objects 1 x batch_sz, accbelief (numActs*numObjs) x batch_sz
        beliefs,actions,objects,accbeliefs = inputs
        self.inputs         = inputs# beliefs, actions, objects
        dim                 = inputs_shape[0][0] 
        assert(inputs_shape[0][1] == inputs_shape[1][1])#batch_size
        assert(inputs_shape[0][1] == inputs_shape[2][1])#batch_size
        assert(inputs_shape[0][1] == inputs_shape[3][1])#batch_size
        assert(inputs_shape[1][0] == 1)#action is a single integer
        assert(inputs_shape[2][0] == 1)#object label is a single integer
        batch_size          = inputs_shape[0][1]
        self.numActions     = int(layer_def.find("numActions").text)
        self.numObjects     = int(layer_def.find("numObjects").text)
        assert(self.numObjects*self.numActions == inputs_shape[3][0])
        assert(self.numObjects == dim)
        #total number of dirichlet units = numActions x numObjects
        num_dirichlets      = self.numObjects * self.numActions
        if clone_from==None:
            self.alphas         = theano.shared( np.random.randint(5,30,[dim,num_dirichlets]).astype(theano.config.floatX)/25.,borrow=True)# dim x num_dirichlets
        else:
            self.alphas     = clone_from.alphas
        #self.alphas         = theano.shared(0.7* np.ones([dim,num_dirichlets]).astype(theano.config.floatX),borrow=True)# dim x num_dirichlets
        #remove 0 from the input belief
        normalized_beliefs  = beliefs + 1.e-6 
        normalized_beliefs  = normalized_beliefs / T.reshape(T.sum(normalized_beliefs,axis=0) , [1,batch_size] )
        log_normed_beliefs  = T.log(normalized_beliefs) # dim x batch_size
        self.log_normed     = log_normed_beliefs

        #calculate Dirichlet probs for the current normalize beliefs
        self.term1          = T.reshape( T.gammaln(T.sum(self.alphas,axis=0)) , [num_dirichlets,1] )
        self.term2          = T.reshape( T.sum(T.gammaln(self.alphas),axis=0) , [num_dirichlets,1] )
        self.term3          = T.dot( T.transpose(self.alphas - 1.) , log_normed_beliefs)# num_dirichlets x batch_size 
        #find a mask based on the actions
        dirichlet_actions   = np.tile(np.arange(self.numActions).reshape([-1,1]),[self.numObjects,1])
        dirichlet_actions   = np.tile(dirichlet_actions,[1,batch_size])
        dirichlet_actions   = theano.shared( dirichlet_actions.astype(theano.config.floatX) , borrow=True)
        in_actions          = T.tile( T.reshape(actions,[1,batch_size]),[num_dirichlets,1])
        self.eq_actions     = T.eq(dirichlet_actions,in_actions)
        #self.current_belief = T.exp(self.term1 - self.term2 + self.eq_actions * self.term3) #this should be normalized for each column
        log_cur_belief      = self.term1 - self.term2 + self.eq_actions * self.term3 #this should be normalized for each column
        #log_cur_belief      = self.term1 - self.term2 + self.term3 #this should be normalized for each column
        log_cur_belief_normd= log_cur_belief - T.reshape( T.max(log_cur_belief,axis=0),[1,batch_size])
        cur_blf             = self.eq_actions * T.exp(log_cur_belief_normd) 
        self.current_belief = cur_blf / T.sum(cur_blf,axis=0)

        acc_is_zero         = T.eq(accbeliefs , 0.)
        accbeliefs_no_0     = acc_is_zero  + (1.-acc_is_zero)*accbeliefs 
        updated_belief      = self.eq_actions * self.current_belief * accbeliefs_no_0 + (1. - self.eq_actions)*accbeliefs# num_dirichlet x batch_size
        sum_up_blf          = T.reshape( T.sum(updated_belief,axis=0) , [1,batch_size] )
        #sum_up_blf_normed   = T.switch( T.eq(sum_up_blf, 0.) , np.ones([1,batch_size]).astype(theano.config.floatX),sum_up_blf)
        #self.updated_belief = updated_belief / sum_up_blf_normed
        self.updated_belief = updated_belief / sum_up_blf
        self.output         = self.updated_belief
        #self.updated_belief = self.current_belief
        
        #construct the outputs 
        # for each class, assign 1s to the components that indicate P(a,o|x)
        #weights_marginalize = np.zeros([self.numObjects,num_dirichlets],dtype=theano.config.floatX)
        #for i in range(self.numObjects):
        #    weights_marginalize[i,i*self.numActions:(i+1)*self.numActions] = 1.
        #weights_margin      = theano.shared( weights_marginalize , borrow=True)
        #self.output         = T.dot( weights_margin, self.updated_belief) 


        #calculating weight updates
        objects_idx         = np.tile( np.arange(self.numObjects).reshape([-1,1]),[1,self.numActions] ).reshape([1,-1])
        objects_idx         = np.tile(objects_idx.reshape([-1,1]), [1,batch_size])# num_dirichlets x batch_size
        objects_idx         = theano.shared(objects_idx.astype(theano.config.floatX), borrow=True)
        in_objects          = T.tile( T.reshape(objects,[1,batch_size]),[num_dirichlets,1])# num_dirichlets x batch_size
        self.idx            = self.eq_actions * T.eq(objects_idx,in_objects)# num_dirichlets x batch_size
        self.idx            = self.idx.astype(theano.config.floatX)
        self.N              = T.reshape(T.sum(self.idx,axis=1),[1,num_dirichlets])
        #take care of 0 in the input to avoid nan in log                                                   
        term5               = T.dot( log_normed_beliefs, T.transpose(self.idx) )#dim x num_dirichlets
        self.update         = self.N * T.reshape(T.psi(T.sum(self.alphas,axis=0)),[1,num_dirichlets]) - self.N * T.psi(self.alphas) + term5
        #self.update         = T.psi(self.alphas) + term5
        
        #calculate log-prob of data                ndirichlets                    ndirichlets
        dir_l_p             = self.N * T.gammaln(T.sum(self.alphas,axis=0)) - self.N * T.sum(T.gammaln(self.alphas),axis=0) + T.sum(term5*(self.alphas-1.),axis=0)
        self.log_p_ao       = T.mean( dir_l_p) 




        self.params         = [self.alphas]        
        self.inputs_shape   = inputs_shape
        #self.output_shape   = [dim,batch_size]
        self.output_shape   = [num_dirichlets,batch_size]

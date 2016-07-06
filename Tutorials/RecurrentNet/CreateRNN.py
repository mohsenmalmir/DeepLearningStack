from __future__ import print_function
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import time
import numpy as np

sys.path.append("../../DeepLearningStack")
import DeepLearningStack
from DeepLearningStack import RecurrentNet



#create a deep network defined in AlexNet.xml
if __name__=="__main__":

    #create deep net
    num_timesteps     = 3 
    print ("Creating Recurrent Net for %d time steps"%num_timesteps)
    config            = "RNNArch.xml"
    #random number generator used to initialize the weights
    rng               = RandomStreams(seed=int(time.time()))
    #for each time step, create a non-recurrent input
    nonrcrnt_ins      = []
    for t in range(num_timesteps):
        in_matrix     = T.matrix("data-step-%d"%t,dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
        nonrcrnt_ins.append( {"data1":in_matrix} )
    #create the recurrent inputs for time step 0
    rct1_step0        = T.matrix("rct1-step-0"%t,dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
    rct2_step0        = T.matrix("rct2-step-0"%t,dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
    fc3_step0         = T.matrix("fc3-step-0"%t,dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
    rcrnt_ins         = {"rct1":rct1_step0,"rct2":rct2_step0,"fc3":fc3_step0}

    #create the graph structure
    rnn               = RecurrentNet.RecurrentNet( rng, nonrcrnt_ins, rcrnt_ins, config, unrolled_len=num_timesteps)

    #draw the RNN
    theano.printing.pydotprint(rnn.name2layer[num_timesteps-1]["act1"], outfile="RNN.png", var_with_name_simple=True)  



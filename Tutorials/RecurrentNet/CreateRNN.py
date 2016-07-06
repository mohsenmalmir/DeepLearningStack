from __future__ import print_function
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys
import time
import numpy as np

sys.path.append("../..")
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
    #size of the input layer is 136x128,
    for t in range(num_timesteps):
        in_matrix     = T.matrix("data-step-%d"%t,dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
        nonrcrnt_ins.append( {"input":[in_matrix,(136,128)]} )#for each input, provide a symbolic variable and its size
    #create the recurrent inputs for time step 0
    #size of the recurrent inputs according to RNNArch.xml (batch size is 128, each size is dim x batch_size)
    # rct1: 128x128
    # rct2: 128x128
    # fc3 : 10x128
    fc1_step0         = T.matrix("fc1-step-0",dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
    fc2_step0         = T.matrix("fc2-step-0",dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
    rcrnt_ins         = {"fc1":[fc1_step0,(128,128)],"fc2":[fc2_step0,(128,128)]}

    #create the graph structure
    rnn               = RecurrentNet.RecurrentNet( rng, nonrcrnt_ins, rcrnt_ins, config, unrolled_len=num_timesteps)

    #create a function for RNN
    
    #inputs to the graph consists of the recurrent inputs for time step 0 and non-recurrent inputs for all time steps
    inputs            = [nonrcrnt_ins[k]["input"][0] for k in range(len(nonrcrnt_ins))]
    for k in rcrnt_ins.keys():
        inputs.append(rcrnt_ins[k][0])
    #outputs of the network include the outputs for each time step
    outputs           = [rnn.name2layer[i]["fc2"].output for i in rnn.name2layer.keys()]
    print("compiling the function")
    f                 = theano.function(inputs=inputs,outputs=outputs)
    #draw the RNN
    graph_img_name    = "RNN.png"
    print("creating graph picture in:",graph_img_name)
    theano.printing.pydotprint(outputs, outfile=graph_img_name, var_with_name_simple=True)  



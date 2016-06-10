import math
import sys
import select
import tty
import termios
import time
import theano
import pprint
import theano.tensor as T
import cPickle
import numpy as np
import scipy.io as sio
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle
import pprint
import copy
import scipy.stats as st

import sys 
sys.setrecursionlimit(50000) 


from DeepNet2 import DeepNet



#create a deep network defined in AlexNet.xml
if __name__=="__main__":

    #create deep net
    print "Creating deep net..."
    config            = "AlexNet.xml"
    #random number generator used to initialize the weights
    rng               = RandomStreams(seed=int(time.time()))
    #create the inputs to the network
    images            = T.tensor4("images")

    #create the graph structure
    net1              = DeepNet(rng,{"image1":images}, config)
                                        


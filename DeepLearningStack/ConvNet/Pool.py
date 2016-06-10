import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

from pylearn2.sandbox.cuda_convnet.filter_acts   import FilterActs
from theano.sandbox.cuda.basic_ops               import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.pool          import MaxPool
from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm

# implementing max pooling
class Pool(object):
    
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a max-pooling layer, that pools across a neighborhood of the inputs 
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type input: tensor.tensor4
            
            :type input_shape: tuple or list of size 4
            :param input_shape: [channels,height,width,batchsize] c01b
            
            :type rs: a random number generator used to initialize weights
            """
        poolDsStride = [ int(layer_def.find("poolds").text),int(layer_def.find("poolstride").text)]
        self.init(input,input_shape,poolDsStride)

    def init(self, input, input_shape, poolDsStride):
    
        self.input               = input
        channels,sz,_,batch_size = input_shape
        output_size              = (sz - poolDsStride[0] + 1)/ (poolDsStride[1]) + (1 if poolDsStride[1]>1 else 0)
        #POOL
        pool_op                  = MaxPool(ds=poolDsStride[0], stride=poolDsStride[1])
        self.output              = pool_op(input)
        self.input_shape         = input_shape
        self.output_shape        = [channels,output_size,output_size,batch_size]
    
        self.params = []













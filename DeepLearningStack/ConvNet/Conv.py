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

# implementing Rectified linear unit
class Conv(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,input,input_shape,rs,clone_from=None):
        """
            Create a (GPU only) convolutional layer with shared variable internal parameters.
            Each filter has a corresponding bias
            
            
            :type layer_def: Element, xml containing configu for Conv layer

            :type input: tensor.tensor4
            
            :type input_shape: tuple or list of size 4
            :param input_shape: [channels,height,width,batchsize] c01b

            :type rs: a random number generator used to initialize weights
        """
        layer_name    = layer_def.attrib["name"]
        convPadStride = [ int(layer_def.find("convpad").text),int(layer_def.find("convstride").text)]
        num_filters   = int(layer_def.find("numfilters").text)
        filter_size   = int(layer_def.find("filtersize").text)
        init_bias     = float(layer_def.find("bias").text)
        rng           = np.random.RandomState(seed=int(time.time()))

        
        self.input    = gpu_contiguous(input)
        image_channels,image_size0,image_size1,batch_size    = input_shape
        filter_shape                              = [image_channels,filter_size,filter_size,num_filters]#c01b
        if clone_from is None:
            #W_bound   = 0.01#numpy.sqrt(6. / (fan_in + fan_out))
            W_bound   = np.sqrt( 2. / (filter_size*filter_size*image_channels) )#initialization from PRELU 
            self.W    = theano.shared( np.asarray(rng.normal(loc=0., scale=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True, name= layer_name+"-W")
            self.b    = theano.shared( np.asarray(init_bias*np.ones((num_filters,)), dtype=theano.config.floatX), borrow=True , name=layer_name+"-b")
        else:
            self.W    = clone_from.W
            self.b    = clone_from.b

        #CONV
        conv_op            = FilterActs(partial_sum=1,pad=convPadStride[0],stride=convPadStride[1])
        contiguous_filters = gpu_contiguous(self.W)
        self.output        = conv_op(self.input, contiguous_filters) + self.b.dimshuffle(0, 'x', 'x','x')

        #output size is equal to (image+2*pad - filter_size + 1) / stride
        output_size0       = (image_size0 + 2 * convPadStride[0] - filter_size + 1 ) / convPadStride[1] + (1 if convPadStride[1]>1 else 0)
        output_size1       = (image_size1 + 2 * convPadStride[0] - filter_size + 1 ) / convPadStride[1] + (1 if convPadStride[1]>1 else 0)
        self.input_shape   = input_shape#[filter_shape[0],img_size,img_size,filter_shape[0]]#c01b
        self.output_shape  = [num_filters, output_size0, output_size1, batch_size]#c01b
        self.params        = [self.W,self.b]













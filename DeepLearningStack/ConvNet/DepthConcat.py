import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# implementing flatten
class DepthConcat(object):
	""" Initialize from xml definition node """
	def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
		self.inputs        = inputs
		n_channels 		   = np.sum([sz[0] for sz in inputs_shape])
		batch_size         = inputs_shape[0][3]
		dim1			   = inputs_shape[0][1]
		dim2               = inputs_shape[0][2]
		for sz in inputs_shape:
			assert(dim1       == sz[1])
			assert(dim2       == sz[2])
			assert(batch_size == sz[3])
		self.output        = T.concatenate(inputs,axis=0)
		self.input_shape   = inputs_shape
		self.output_shape  = [n_channels,dim1,dim2,batch_size]
		self.params        = []













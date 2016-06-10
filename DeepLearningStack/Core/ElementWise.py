import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T


# element-wise operations on the input maps 
class ElementWise(object):
    """ Initialize from xml definition node """
    def __init__(self,layer_def,inputs,inputs_shape,rs,clone_from=None):
        """
            Create an elementwise operation layer
            This layer can be used to add or multiply different inputs across a specific dimension
            The supported operations are SUM for addition and PROD for multiplication.
            
            :type layer_def: Element, xml containing configu for Conv layer
            
            :type inputs: list of inputs, all should have the same size 
            
            :type inputs_shapes: list of size of inputs 
            
            :type rng: a random number generator used to initialize weights
            """
        self.inputs        = inputs
        assert(len(inputs)>1)
        #make sure all inputs have the same size
        for sh in inputs_shape[1:]:
            assert( np.all( np.asarray(sh) == inputs_shape[0]) )

        self.op            = layer_def.find("operation").text

        #calculate the output
        temp               = self.inputs[0]
        if self.op=="SUM":
            for i in inputs[1:]:
                temp = temp + i
        elif self.op=="PROD":
            for i in inputs[1:]:
                temp = temp * i
        else:
            assert(False)

        self.output        = temp 
        self.input_shape   = inputs_shape
        self.output_shape  = inputs_shape[0] 
        self.params        = []













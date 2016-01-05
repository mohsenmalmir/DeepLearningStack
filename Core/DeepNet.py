import os
import sys
import time
import time
import pprint
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


import xml.etree.ElementTree as ET

from Img                    import Img
from Data                   import Data
from Conv                   import Conv
from Flatten                import Flatten
from LRN                    import LRN
from LU                     import LU
from Pool                   import Pool
from Rectifier              import Rectifier
from Softmax                import Softmax
from Normalize              import Normalize
from Dropout                import Dropout
from ActionSelection        import ActionSelection
from Concatenate            import Concatenate
from Sigmoid                import Sigmoid
from Tanh                   import Tanh
from NaiveBayesBeliefUpdate import NaiveBayesBeliefUpdate
from DepthConcat			import DepthConcat
from DirichletLayer         import DirichletLayer

#maps type names into classes
type2class      = {"Data":Data, "Conv":Conv, "Flatten":Flatten,"LRN":LRN,"LU":LU,"Pool":Pool,
                    "Rectifier":Rectifier,"Softmax":Softmax, "Image":Img, "Dropout":Dropout,
                    "ActionSelection":ActionSelection, "Concatenate":Concatenate,
                    "Sigmoid":Sigmoid, "Tanh":Tanh, "Normalize":Normalize,
                    "NaiveBayesBeliefUpdate":NaiveBayesBeliefUpdate,
					"DepthConcat":DepthConcat,
                    "DirichletLayer":DirichletLayer,
                  }
# Stack of ReLU followed by LU
class DeepNet(object):

    def __init__(self, rng, input, configFile, clone_from=None):
        """Initialize the parameters for the Deep Net

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.tensor4
        :param input: symbolic variable that represents image batch

        :type configFile: filename containing the network architecture
        :param image_shape: string

        """
        self.supplied_inputs      = input#dict of name:symvar
        self.output_dims          = dict()#dictionary of inp:size for the input
        for inp_name in input.keys():
            self.output_dims[inp_name] = []
        tree                      = ET.parse(configFile)
        root                      = tree.getroot()
        layers_def                = []
        for layer in root:
            layers_def.append(layer)
        #building network DAG
        netbuilt         = False
        self.layers      = []
        self.name2layer  = dict()
        self.params      = []
        while not netbuilt:
            for layer in layers_def:
                layer_name,layer_type = layer.attrib["name"],layer.attrib["type"]
                layer_input           = layer.find("input").text if layer.find("input") is not None else None
                if layer_type in ["Concatenate","DepthConcat","NaiveBayesBeliefUpdate","DirichletLayer"]:
                    #check for all inputs
                    inputs           = layer.findall("input")
                    inputs_text      = [inp.text for inp in inputs]
                    input_satiesfied = [inp.text in self.supplied_inputs.keys() for inp in inputs]
                    if np.all(input_satiesfied):
                        inputs       = [self.supplied_inputs[inp] for inp in inputs_text]
                        output_dims  = [self.output_dims[inp] for inp in inputs_text]
                        if clone_from!=None:#if cloning, then initialize this layer from the clone
                            newLayer    = type2class[layer_type](layer,inputs,output_dims,rng,clone_from.name2layer[layer_name])
                        else:
                            newLayer    = type2class[layer_type](layer,inputs,output_dims,rng)
                        self.layers.append(newLayer)#create layer from xml definition
                        self.name2layer[layer_name]          = newLayer
                        self.params                         += newLayer.params
                        self.supplied_inputs[layer_name]     = newLayer.output
                        self.output_dims[layer_name]         = newLayer.output_shape
                        layers_def.remove(layer)
                elif layer_input not in self.supplied_inputs.keys():
                    continue
                else:
                    print "creating layer:",layer_name, "with input: ",layer_input
                    #create the layer, add it to self.layers
                    #each layer is initialized by its definition from xml, its input variable, the dimensions of its input and rs
                    if clone_from!=None:#if cloning, then initialize this layer from the clone
                        newLayer                         = type2class[layer_type](layer,self.supplied_inputs[layer_input],self.output_dims[layer_input],rng,clone_from.name2layer[layer_name])
                    else:
                        newLayer                         = type2class[layer_type](layer,self.supplied_inputs[layer_input],self.output_dims[layer_input],rng)
                    self.layers.append(newLayer)#create layer from xml definition
                    self.name2layer[layer_name]          = newLayer
                    self.params                         += newLayer.params
                    self.supplied_inputs[layer_name]     = newLayer.output
                    self.output_dims[layer_name]         = newLayer.output_shape
                    layers_def.remove(layer)
            if len(layers_def)==0:
                netbuilt = True
        print "network built!"












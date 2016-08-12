from __future__ import print_function
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

from ..Input.Img                    import Img
from ..Input.Data                   import Data
from ..ConvNet.Conv                 import Conv
from .Flatten                       import Flatten
from ..ConvNet.LRN                  import LRN
from .LU                            import LU
from ..ConvNet.Pool                 import Pool
from .Rectifier                     import Rectifier
from .Softmax                       import Softmax
from ..ConvNet.Normalize            import Normalize
from .Dropout                       import Dropout
from ..RL.ActionSelection           import ActionSelection
from .Concatenate                   import Concatenate
from .Sigmoid                       import Sigmoid
from .Tanh                          import Tanh
from ..RL.NaiveBayesBeliefUpdate    import NaiveBayesBeliefUpdate
from ..ConvNet.DepthConcat			import DepthConcat
from ..RL.DirichletLayer            import DirichletLayer
from ..ConvNet.BatchNormalize       import BatchNormalize
from .ElementWise                   import ElementWise
from ..RL.Gaussian                  import Gaussian
from ..RL.GaussianObs               import GaussianObs
from ..Mem.LSTM                     import LSTM 

#maps type names into classes
type2class      = {"Data":Data, "Conv":Conv, "Flatten":Flatten,"LRN":LRN,"LU":LU,"Pool":Pool,
                    "Rectifier":Rectifier,"Softmax":Softmax, "Image":Img, "Dropout":Dropout,
                    "ActionSelection":ActionSelection, "Concatenate":Concatenate,
                    "Sigmoid":Sigmoid, "Tanh":Tanh, "Normalize":Normalize,
                    "NaiveBayesBeliefUpdate":NaiveBayesBeliefUpdate,
					"DepthConcat":DepthConcat,
                    "DirichletLayer":DirichletLayer,
                    "BatchNormalize":BatchNormalize,
                    "ElementWise":ElementWise,
                    "Gaussian":Gaussian,
                    "GaussianObs":GaussianObs,
                    "LSTM":LSTM,
                  }
# Stack of ReLU followed by LU
class FeedForwardNet(object):

    def __init__(self, rng, inputs, configFile, clone_from=None):
        """Initialize the parameters for the Deep Net

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: list of dictionaries 
        :param input: this is a list of (symvar, size) for the inputs to the network 

        :type configFile: filename containing the network architecture
         

        :type clone_from: A computation graph to clone from 
        :param clone_from: This graph should contain all the weights, from which the current network will be initialized.
        :                   This is useful in cases such as transferring the weights to a different architecture that shares some layers  
        """
        self.supplied_inputs      = dict()#dict of name:symvar
        self.output_dims          = dict()#dictionary of inp:size for the input
        for inp_name in inputs.keys():
            self.supplied_inputs[inp_name]  = inputs[inp_name][0] 
            self.output_dims[inp_name]      = inputs[inp_name][1] 
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
        self.tied        = dict()
        while not netbuilt:
            layer_added = False#for debugging network structure
            for layer in layers_def:
                #find the the layer's name and type
                layer_name,layer_type = layer.attrib["name"],layer.attrib["type"]
                layer_inputs          = layer.findall("input")#layer.find("input").text if layer.find("input") is not None else None
                #indicates if the parameters have to be the same as some other layer
                tie_from              = layer.find("tie").text if layer.find("tie") is not None else None
                #check if all the inputs to the network are already supplied
                inputs_text           = [inp.text for inp in layer_inputs]
                input_satiesfied      = [inp.text in self.supplied_inputs.keys() for inp in layer_inputs]
                #if a layer's params are tied to another layer, make sure the first layer is already created
                if tie_from != None and tie_from not in self.name2layer.keys():
                    continue
                #if layer_type in ["Concatenate","DepthConcat","NaiveBayesBeliefUpdate","DirichletLayer","ElementWise","GaussianObs"]:
                #check for all inputs
                if np.all(input_satiesfied):
                    print ("creating layer:",layer_name)
                    symvar_inputs     = [self.supplied_inputs[inp] for inp in inputs_text]
                    symvar_sizes      = [self.output_dims[inp] for inp in inputs_text]
                    #this is to make sure that layers with 1 input will not receive a list variable
                    if len(symvar_inputs)==1:
                        symvar_inputs = symvar_inputs[0]
                        symvar_sizes  = symvar_sizes[0]
                    #if cloning and the layer is found in the source network, then initialize this layer from the clone
                    if clone_from!=None and (layer_name in clone_from.name2layer.keys()):
                        newLayer              = type2class[layer_type](layer,symvar_inputs,symvar_sizes,rng,clone_from=clone_from.name2layer[layer_name])
                        self.tied[layer_name] = clone_from.tied[layer_name]#this layer is cloned  
                    #else if it is tied, then create it using the tied network
                    elif tie_from!=None and (layer_name in self.name2layer.keys()):#if the parameters are tied to gether
                        newLayer              = type2class[layer_type](layer,symvar_inputs,symvar_sizes,rng,clone_from=self.name2layer[tie_from])
                        self.tied[layer_name] = True 
                    #otherwise simply create it with regular initialization of parameters
                    else:
                        newLayer              = type2class[layer_type](layer,symvar_inputs,symvar_sizes,rng)
                        self.tied[layer_name] = False 
                        self.params           += newLayer.params

                    self.layers.append(newLayer)#create layer from xml definition
                    self.name2layer[layer_name]          = newLayer
                    if type(newLayer.output)==dict:
                        self.supplied_inputs[layer_name]     = newLayer.output[layer_name]
                        self.output_dims[layer_name]         = newLayer.output_shape[layer_name]
                    else:
                        self.supplied_inputs[layer_name]     = newLayer.output
                        self.output_dims[layer_name]         = newLayer.output_shape
                    layers_def.remove(layer)
                    layer_added                          = True
            if len(layers_def)==0:
                netbuilt = True
            if layer_added==False:
                print("Error: Can't add any new layer to the network!")
                print("Please check network structure for incorrect links and non-existing inputs.")
                print("Here is a list of correctly created layers:")
                pprint.pprint(self.output_dims)
                assert(False)
        print ("network built!")












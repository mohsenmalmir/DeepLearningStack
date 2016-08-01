from __future__ import print_function
import os
import sys
import time
import time
import pprint
import copy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


import xml.etree.ElementTree as ET

from ..Input.Img                    import Img
from ..Input.Data                   import Data
from ..ConvNet.Conv                 import Conv
from ..Core.Flatten                 import Flatten
from ..ConvNet.LRN                  import LRN
from ..Core.LU                      import LU
from ..ConvNet.Pool                 import Pool
from ..Core.Rectifier               import Rectifier
from ..Core.Softmax                 import Softmax
from ..ConvNet.Normalize            import Normalize
from ..Core.Dropout                 import Dropout
from ..RL.ActionSelection           import ActionSelection
from ..Core.Concatenate             import Concatenate
from ..Core.Sigmoid                 import Sigmoid
from ..Core.Tanh                    import Tanh
from ..RL.NaiveBayesBeliefUpdate    import NaiveBayesBeliefUpdate
from ..ConvNet.DepthConcat			import DepthConcat
from ..RL.DirichletLayer            import DirichletLayer
from ..ConvNet.BatchNormalize       import BatchNormalize
from ..Core.ElementWise             import ElementWise
from ..RL.Gaussian                  import Gaussian
from ..RL.GaussianObs               import GaussianObs
from ..Mem.LSTM                     import LSTM 
from ..Mem.GRU                      import GRU 

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
                    "GRU":GRU,
                  }
class RecurrentNet(object):
    """
        This is a class for creating generic recurrent neural networks such as LSTM, RNN etc.
        The key idea in this network is that each layer can receive recurrent input from any 
        layer or provide recurrent input to any layer of the network.
        Definition of network architecture is done using XML files, but the difference with
        FeedforwardNet is that layers can use 'RecurrentIn' and 'RecurrentOut' tags to declare
        the recurrent inputs or outputs of the network.
    """

    def __init__(self, rng, nonrcrnt_inputs, rcrnt_inputs, configFile, clone_from=None, unrolled_len=1):
        """Initialize the parameters for the recurrent net

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type nonrcrnt_inputs: list of dictionaries of name:symvar 
        :param nonrcrnt_inputs: this is a lsit of dictionaries, each of which that specifies the non-recurrent inputs, the ones that are provided to the network
        :                       from outside. These non-recurrent inputs will be fed to the network at each time step, the length of the list should be equal
                                to the unrolled_len parameter. 


        :type rcrnt_inputs: dictionary of the form name:symvar
        :param rcrnt_inputs: This dictionary indicates the symbolic variables that are used as recurrent inputs for time step 0, that is when there are no
                              recurrent inputs from the network. Note that there are only one set of such variables, and it is used to feed the recurrent
                              variables in the network at time 0. 

        :type configFile: filename containing the network architecture
         

        :type clone_from: A computation graph to clone from 
        :param clone_from: This graph should contain all the weights, from which the current network will be initialized.
        :                   This is useful in cases such as transferring the weights to a different architecture that shares some layers  

        :type unrolled_len: integer
        :param unrolled_len: Indicates the number of times that the network shoud be unrolled.
        :                     if the nework is unrolled 4 times, then the input to the network
        :                     should be an array of length 4, one for each time step.
        """

        assert( unrolled_len == len(nonrcrnt_inputs))#make sure that the non-recurrent inputs are provided for the entire steps of unrolled network
        #parse the XML tree
        tree                      = ET.parse(configFile)
        root                      = tree.getroot()
        layers_def                = []
        for layer in root:
            layers_def.append(layer)


        #unroll the time step 0
        layers,name2layer,params,output_dims,rcrnt_output,tied = self.UnrollOneStep(rng, layers_def, nonrcrnt_inputs[0], rcrnt_inputs)
        #store the layers in the object instance
        #layers and name2layer are now dictionaries from timestep to variables
        self.layers               = {0:[]}#each time step is an array
        self.name2layer           = { 0:{} }#each time step is a dictionary
        self.params               = params#the parameters are only from the time step 0, since the rest of time steps are tied to timestep 0 
        self.output_dims          = output_dims#this is only for time step 0, since the rest of time steps are copies of time step 0
        for k in name2layer.keys():
            self.layers[0].append(name2layer[k])
            self.name2layer[0][k] = name2layer[k]
        self.tied                 = tied#this is just for time step 0, since all the parameters in time steps > 0 are tied to time step 0
        for i in range(1,unrolled_len):
            #unroll the time step i
            layers,name2layer,params,output_dims,rcrnt_output,tied = self.UnrollOneStep(rng, layers_def, nonrcrnt_inputs[i], rcrnt_output, self.name2layer[i-1])
            #store the layers in the object instance
            self.layers[i]        = []
            self.name2layer[i]    = dict() 
            for k in name2layer.keys():
                self.layers[i].append(name2layer[k])
                self.name2layer[i][k] = name2layer[k]


    def UnrollOneStep(self, rng, layers_def, nonrcrnt_inputs, rcrnt_inputs, net_prev_timestep = None):
        """
            This function unrolls a network for one step, e.g. creates a network graph
            that receives recurrent and non-recurrent inputs and produces the recurrent
            and non-recurrent outputs of the network for one step, e.g. one pass through
            the network.
        
            :type rng: theano.tensor.shared_randomstreams.RandomStreams
            :param rng: random number generator for weights initialization

            :type layers_def: list of xml.etree.ElementTree.Element
            :param layers_def: this is the parsed tree of the network's definition XML file
     
            :type nonrcrnt_inputs: dictionary of name:(symvar,size) 
            :param symvars: a dictionary of symbolic variables and their size, referred to by the name. 
            :               The name comes from the network definition XML file.
            :               These are the inputs that are provided from external source, e.g. not from the network's output


            :type rcrnt_inputs: dictionary of name:(symvar,size) 
            :param rcrnt_inputs: This is a dictionary of recurrent inputs to the network. It indicates the name and size of 
            :                    symbolic variables that are produced by the network at time step t-1, where t is the current time step 

            :type net_prev_timestep: dictionary of name:symvar 
            :param net_prev_timestep: this is the network created for the previous time step. It is used to tie the weights for current time step. 
        """
        #building network 
        supplied_inputs  = copy.copy(nonrcrnt_inputs)#non-recurrent inputs, i.e. inputs from the current time step
        layers_def       = copy.copy(layers_def)
        netbuilt         = False
        layers           = []
        name2layer       = dict()
        params           = []
        tied             = dict()
        rcrnt_output     = dict()
        output_dims      = dict()
        while not netbuilt:
            layer_added  = False
            for layer in layers_def:
                #find the the layer's name and type
                layer_name,layer_type        = layer.attrib["name"],layer.attrib["type"]
                recrnt_ins                   = layer.findall("recurrent_input") 
                nonrecrnt_ins                = layer.findall("input")
                #indicates if the parameters have to be the same as some other layer
                tie_from                     = layer.find("tie").text if layer.find("tie") is not None else None
                #check if all the inputs to the network are already supplied
                nonrecrnt_ins_satiesfied     = [inp.text in supplied_inputs.keys() for inp in nonrecrnt_ins]
                rcrnt_ins_satiesfied         = [inp.text in rcrnt_inputs.keys() for inp in recrnt_ins]
                #if a layer's params are tied to another layer, make sure the first layer is already created
                if tie_from != None and tie_from not in name2layer.keys():
                    continue
                #check if all recurrent and non-recurrent inputs are satisfied 
                if np.all(nonrecrnt_ins_satiesfied) and np.all(rcrnt_ins_satiesfied): 
                    print ("creating layer:",layer_name)
                    layer_added       = True
                    #arrange all the inputs in the order that is specified in the XML,
                    symvar_inputs     = []
                    symvar_sizes      = []
                    for elmnt in layer:
                        if elmnt.tag == "input":
                            symvar_inputs.append( supplied_inputs[elmnt.text][0] )
                            symvar_sizes.append(  supplied_inputs[elmnt.text][1])
                        elif elmnt.tag=="recurrent_input":
                            symvar_inputs.append( rcrnt_inputs[elmnt.text][0] )
                            symvar_sizes.append(  rcrnt_inputs[elmnt.text][1])
                    #this is to make sure that layers with 1 input do not receive a list of 1 item
                    if len(symvar_inputs)==1:
                        symvar_inputs = symvar_inputs[0]
                        symvar_sizes  = symvar_sizes[0]
                    #print("layer sizes:",symvar_sizes)
                    #priority of weight cloning: clone_from > tie > normal
                    #if tying from the previous time step 
                    if net_prev_timestep!=None and (layer_name in net_prev_timestep.keys()):
                        newLayer              = type2class[layer_type](layer,symvar_inputs,symvar_sizes,rng,clone_from=net_prev_timestep[layer_name])
                        tied[layer_name]      = True#if this is a copy from the previous time step, then it is tied 
                    #tying from the current time step
                    elif tie_from!=None and (layer_name in name2layer.keys()):#if the parameters are tied to gether
                        newLayer              = type2class[layer_type](layer,symvar_inputs,symvar_sizes,rng,clone_from=name2layer[tie_from])
                        tied[layer_name]      = True#if this is a tie from current time step, 
                    #otherwise simply create it with regular initialization of parameters
                    else:
                        newLayer    = type2class[layer_type](layer,symvar_inputs,symvar_sizes,rng)
                        tied[layer_name]      = False 
                        params               += newLayer.params
                    layers.append(newLayer)#create layer from xml definition
                    name2layer[layer_name]          = newLayer
                    #if output is a dictionary, i.e. there are multiple outputs, then the default output is the one with the layer name
                    if type(newLayer.output)==dict:
                        supplied_inputs[layer_name]     = (newLayer.output[layer_name], newLayer.output_shape[layer_name])
                        output_dims[layer_name]         = newLayer.output_shape[layer_name]
                    else:#if it has a single output, then use that as the default output
                        supplied_inputs[layer_name]     = (newLayer.output, newLayer.output_shape)
                        output_dims[layer_name]         = newLayer.output_shape
                    #also look into <output> tags, as they provide further outputs
                    multiouts                           = layer.findall("output")
                    for out in multiouts:
                        #for multioutputs, add each one to the list of supplied inputs regardless of their feedback property
                        #the ambiguity between recurrent and non-recurrent inputs is resolved in the input vs. recurrent input tag
                        supplied_inputs[out.text]   = [newLayer.output[out.text], newLayer.output_shape[out.text]]
                        output_dims[out.text]           = newLayer.output_shape[out.text]
                        #multiple, recurrent output
                        if out.attrib["feedback_output"].lower()=="yes":
                            rcrnt_output[out.text]      = (newLayer.output[out.text], newLayer.output_shape[out.text])
                    #the default output of the layer
                    if "feedback_output" in layer.attrib.keys():
                        if layer.attrib["feedback_output"].lower()=="yes":
                            if type(newLayer.output)==dict:
                                rcrnt_output[layer_name] = (newLayer.output[layer_name], newLayer.output_shape[layer_name])
                            else:#if it has a single output, then use that as the default output
                                rcrnt_output[layer_name] = (newLayer.output, newLayer.output_shape)
                    layers_def.remove(layer)
            if len(layers_def)==0:
                netbuilt = True
            elif layer_added==False:
                #if no layer was added, there is some connection error in the network's defintion file
                # should check the xml file for non-existing inputs
                print("Error: Can't add any new layer to the network!")
                print("Please check network structure for incorrect links and non-existing inputs.")
                print("Here is a list of correctly created layers:")
                pprint.pprint(output_dims)
                print("Here is the list of layers that are not created yet:")
                pprint.pprint([ld.attrib["name"] for ld in layers_def])
                assert(False) 

        print ("network built!")
        #print("recurrent outputs:")
        #pprint.pprint(rcrnt_output)
        return layers,name2layer,params,output_dims,rcrnt_output,tied
       









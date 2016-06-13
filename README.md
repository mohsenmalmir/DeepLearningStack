# DeepLearningStack
DeepLearningStack allows easy definition of deep neural networks with XML configuration files.
The package contains modules for definition of MLP, convolutional networks, reinforcement learning and recurrent neural networks.

A library for deep learning in Thenao. It allows easy definition of deep networks using XML configuration files, and has different packages for reinforcement learning and convolutional networks.

#Installation
Download the package, cd into the directory and type:
```
sudo python setup.py install
```

Alternatively, you can use pip:
```
pip install DeepLearningStack
```

#Tutorials
This tutorial shows how to create a simple convnet using an XML configuration file. The network is defined in *AlexNet.xml*, which defines different layers and connect them to each other to form the computation graph of the network. Each layer is deined as
```
<layer type="" name="">
<input>input1_layer_name</input>
<input>input2_layer_name</input>
...
<param1>value1</param1>
<param2>
  <param2_tag1>val11<param2_tag1>
  ...
</param2>
...

</layer>
```
Each layer should have a type, e.g. convolution, and an arbitrary but unique name. The name of the layer is used to direct its output to another layer through the ```<input>``` tag. Each layer type has layer-specific parameters, such as kernel size for convolution layer. See *config/Layers.xml* for a complete list of layers and their parameters.

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
#AlexNet
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
#Deep Q-learning
This tutorial implements the deep Q-learning method for active object recognition described in [1]. Active object recognition deals with smart exploration of objects to achieve higher accuracies with smaller number of images processed. A neural network is trained, using Q-learning iterative update rule, to perform a series of actions on GEMRS [2] objects to increase the accuracy of label prediction. The architecture of the network defined in ```DQLArch.xml``` is composed of two fully connected layers followed by softmax, which predicts the action-values for 10 different actions. The actions rotate the objects inward or outward w.r.t camera with different magnitudes.

To run this tutorial, you need to download the GERMS dataset from [3]. These are the belief vectors over different object labels, calculated from features extracted by VGG 7-layer network. This network was pre-trained on ImageNet, and was not fine-tuned for GERMS. After downloading the belief files, put them in ```Tutorials/DeepQL/``` and execute ```DQL.py``` script.







[1] Malmir M, Sikka K, Forster D, Movellan J, Cottrell GW. Deep Q-learning for Active Recognition of GERMS: Baseline performance on a standardized dataset for active learning. InProceedings of the British Machine Vision Conference (BMVC), pages 2016 Apr 13 (pp. 161-1).

[2] http://rubi.ucsd.edu/GERMS

[3] https://drive.google.com/folderview?id=0BxZOUQHBUnwmQUdWRGlPMGw4WHM&usp=sharing

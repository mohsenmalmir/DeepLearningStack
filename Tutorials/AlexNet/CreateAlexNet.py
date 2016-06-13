from __future__ import print_function
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import sys
sys.path.append("../../")
import time

import DeepLearningStack
from DeepLearningStack import FeedForwardNet



#create a deep network defined in AlexNet.xml
if __name__=="__main__":

    #create deep net
    print ("Creating deep net graph")
    config            = "AlexNet.xml"
    #random number generator used to initialize the weights
    rng               = RandomStreams(seed=int(time.time()))
    #create the inputs to the network
    images            = T.tensor4("images")
    lr		          = T.scalar("lr")#learning rate for weight update, defined as sym var to be updatable during training

    #create the graph structure
    net1              = FeedForwardNet.FeedForwardNet(rng,{"input":images}, config)



    #create the cost
    print ("Creating the cross-entropy cost function")
    size_minibatch    = net1.name2layer["image1"].output_shape[-1]#c01b
    target_labels     = theano.shared(np.zeros([size_minibatch],dtype=np.int32),borrow=True)
    cost_clf          = T.mean(T.nnet.categorical_crossentropy(net1.name2layer["softmax"].output.T, target_labels))#cross entropy classification cost

    #creating the updates, gradient descent with momentum
    updates          = []
    mmnt             = []

    #for each layer, calculate the gradient
    print ("Creating updates for network weights using GD and mometum")
    alpha            = 0.9 #momentum coefficient
    weight_decay     = 0.0005 #weight decay for regularization
    for name in net1.name2layer.keys():
        if net1.tied[name] == False:#only if the parameters are not tied, update them
            for param in net1.name2layer[name].params:
                grad_clf = T.grad(cost_clf,param)#derivative of the classification cost w.r.t param
                mmnt.append(theano.shared(np.zeros_like(param.get_value())))
                updates.append((mmnt[-1], alpha * mmnt[-1] + (1.-alpha) * lr * grad_clf ) )
                updates.append((param, param - mmnt[-1] - weight_decay * param) )

                                        
    #create the train function
    print ("creating train function")
    f_train      = theano.function(inputs=[images,lr],outputs=[
                                                       cost_clf,
                                                       net1.name2layer["softmax"].output,
                                                      ],
                                    givens={},
                                    updates = updates,
                              )




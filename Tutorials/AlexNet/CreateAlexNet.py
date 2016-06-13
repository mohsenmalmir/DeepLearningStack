import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import ..DeepLearningStack



#create a deep network defined in AlexNet.xml
if __name__=="__main__":

    #create deep net
    print "Creating deep net..."
    config            = "AlexNet.xml"
    #random number generator used to initialize the weights
    rng               = RandomStreams(seed=int(time.time()))
    #create the inputs to the network
    images            = T.tensor4("images")

    #create the graph structure
    net1              = FeedForwardNet.FeedForwardNet(rng,{"image1":images}, config)
                                        


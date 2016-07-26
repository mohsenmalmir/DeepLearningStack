from __future__ import print_function




class RBM:
    def __init__(self, rng, configFile, cloneFrom=None)
        """
        initializing the RBM from the specified configFile

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type configFile: string
        :param configFile: the configuration file that speicifes the details of RBM net 

        :type cloneFrom: RBM object 
        :param cloneFrom: RBM from which the weights of this network will be cloned

        """
        #read the layers sizes, and neuron types
        pass

    def CD(self, n, v):
        """
            contrastive divergence for n-steps. This funciton performs Gibbs sampling on the network

            :type n: int
            :param n: specifies the number of steps in the CD(n)

            :type v: numpy matrix, 2D
            :param v: specifies the value of visible units at the start of sampling
        """
        #perform sampling, return v_n, h_n


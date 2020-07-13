#
# layer.py
# ~~~~~~~~

import numpy as np

class Layer(object):

    """ Initialize a hidden layer of a neural network.

        Parameters
        ----------

        depth : {int}
            Defines the depth of the current layer.

        n_inputs : {int}
            The number of inputs in each neuron of the layer.

        n_neurons : {int}
            The number of neurons in the layer.

        bias : {float}, default=0
            The bias of the layer.

        activation : {string}, default="sigmoid"
            The activation fucntion used at the layer

        state : {string}, default="hideden"
    """

    def __init__(self,
                 depth,
                 n_inputs,
                 n_neurons,
                 bias=0,
                 activation="sigmoid",
                 state="hidden"):
        self.depth = depth
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.bias = bias * np.ones((self.n_neurons, 1))
        self.activation = activation
        self.state = state
        self.z = None
        self.a = None
        self.delta = None
        self.delta_w = None
        self.delta_b = None
        np.random.seed(0)
        self.weights = np.random.randn(self.n_neurons, self.n_inputs) / np.sqrt(self.n_inputs)

    def __repr__(self):
        return """Layer(depth={},
      n_inputs={},
      n_neurons={},
      bias={},
      activation={},
      state={})""".format(self.depth, self.n_inputs, self.n_neurons, \
                          self.bias[0][0], self.activation, self.state)

    def __call__(self, inputs):
        inputs = inputs.reshape((self.n_inputs, 1))
        self.z = np.dot(self.weights, inputs) + self.bias
        self.a = self.transfer(self.z)
        return self.a
    
    def transfer(self, x):
        #return np.log(1 + np.exp(x))
        return 1 / (1 + np.exp(-x))
        #return np.tanh(x)
        
    def transfer_derivative(self, x):
        #return 1 / (1 + np.exp(-x))
        return np.array(self.transfer(x) * (1.0 - self.transfer(x)))
        #return 1 - np.tanh(x)**2

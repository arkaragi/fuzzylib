# "
# ann_mnist.py
# ~~~~~~~~~

# Standard libraries
import csv
import math
import random
import numpy as np
import pandas as pd

# External libraries
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from layer import Layer
from sklearn.metrics import r2_score

class NeuralNetwork(object):

    """ Base class for a multilayer neural network.

        Parameters
        ----------

        n_inputs : {int}
            The number of features of each sample.

        n_hidden : {list}
            This parameter defines the number of hidden layers inside
            the network and the number of neurons inside each layer.
            A multilayer ANN with len(n_hidden) layers will be initialized.
            Each layer will contain n_hidden[j] neurons, where
            0 <= j <= len(n_hidden)-1.
                            
       n_outputs : {int}
            The number of neurons at the output layer. In case of
            classification, n_outputs equals to the number of distinct
            classes. In case of regression, n_outputs equals to the
            number of predicted values.

        eta : {float}, default=0.1
            The learning parameter of a neural network. Determines the the
            change in the weights after back-propagation.

        bias : {float, list}, default=0.
            The initial bias at each layer of the network, including the
            output layer.

            if {float}, each layer is biased by the same ammount of bias
            if {list}, every layer is biased by an ammount equal to bias[j].
            The length of the list must be equal to n_hidden + n_outputs.
    """

    def __init__(self,
                 n_inputs,
                 n_hidden,
                 n_outputs,
                 eta=0.1,
                 momentum=0.,
                 bias=0.,
                 activation="sigmoid",
                 debug=False):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.eta = eta
        self.momentum = momentum
        self.bias = bias
        self.activation = activation
        self.debug = False
        self.initialize_network()

    def initialize_network(self):
        """Initializes the layers of the network."""
        self.network = list()
        self.n_layers = len(self.n_hidden)
        for j in range(self.n_layers):
            if j == 0:
                self.network.append(Layer(j, self.n_inputs, self.n_hidden[j], bias=self.bias, activation=self.activation))
            else:
                self.network.append(Layer(j, self.n_hidden[j-1], self.n_hidden[j], bias=self.bias, activation=self.activation))
        else:
            j = len(self.network)
            self.network.append(Layer(j, self.n_hidden[j-1], self.n_outputs, bias=self.bias, activation=self.activation, state="output"))
                
    def train_network(self, train, n_epoch):
        """Train a network for a fixed number of epochs."""
        for epoch in range(n_epoch):
            self.model_error = 0
            for row in train:
                self.feed_forward(row)
                self.delta_output(row)
                self.backward_propagate_error(row)
                self.update_weights(row)
            self.model_error /= 2*len(train)
            if epoch%100 == 0:
                print('>epoch={}, lrate={}, error={:.3e}'.format(epoch, self.eta, float(self.model_error[0])))

    def feed_forward(self, row):
        """Forward propagate input to a network output."""
        inputs = row[:-1]
        for layer in self.network:
            outputs = layer(inputs)
            inputs = outputs
        self.outputs = outputs

    def delta_output(self, row):
        self.y = row[-1]
        self.model_error += (self.y-self.outputs)**2

    def backward_propagate_error(self, row):
        """Backpropagate error and store in neurons."""
        for layer in reversed(self.network):
            if layer.state == 'output':
                delta = self.y - layer.a
            else:
                delta = np.dot(self.network[layer.depth+1].weights.transpose(), self.network[layer.depth+1].delta)
            layer.delta = delta * layer.transfer_derivative(layer.z)
            
    def update_weights(self, row):
        """Update network weights with error."""
        inputs = np.array([[r] for r in row[:-1]])
        self.prev_delta = []
        for j, layer in enumerate(self.network):            
            if j != 0: inputs = self.network[j-1].a
            delta_w = self.eta * np.dot(layer.delta, inputs.transpose())
            delta_b = self.eta * layer.delta
            if layer.delta_w is None:
                layer.weights = layer.weights + delta_w
                layer.bias = layer.bias + delta_b
            else:
                layer.weights = layer.weights + delta_w + (self.momentum * layer.delta_w)
                layer.bias = layer.bias + delta_b
            layer.delta_w = delta_w
            layer.delta_b = delta_b

    def predict(self, row):
        """Make a prediction with a network."""
        self.feed_forward(row)
        return self.outputs


##############################################################################


# Initalize train dataset
scaler = MinMaxScaler()
db = pd.read_csv('o3db_train.csv')
db = db.dropna(subset=['Target'])
idata = db.to_numpy()
X = idata[1:, 3:-1]
y = idata[1:, -1]
M, N = X.shape
c = 1               
var_names = list(db)[3:]
dataset = [np.append(X[i], y[i]) for i in range(M)]
scaler.fit(dataset)
train_dataset = scaler.transform(dataset)
print("train dataset initialized...")

# Initialize test dataset
db_test = pd.read_csv('o3db_test.csv')
db_test = db_test.dropna(subset=['Target'])
tdata = db_test.to_numpy()
Xt = tdata[1:, 3:-1]
yt = tdata[1:, -1]
Mt, Nt = Xt.shape
print("test dataset initialized...")
dataset = [np.append(Xt[i], yt[i]) for i in range(Mt)]
test_dataset = scaler.transform(dataset)

# Initialize and train a neural network
n_hidden = [10, 10]
epochs = 1001
net = NeuralNetwork(N, n_hidden, c, eta=0.1, momentum=0.9, bias=0.01)
net.train_network(train_dataset, epochs)

### Calculte network accuracy based on training samples
##train_predictions = []
##for row in train_dataset:
##    train_predictions.append(net.predict(row))
##train_predictions = np.array(train_predictions).flatten()
##train_error = []
##for a, b in zip(train_predictions, train_dataset[:,-1]):
##    train_error.append((a-b)**2)
##train_error = np.array(train_error).flatten()
### plot the training predictions
##plt.figure(1)
##plt.plot(train_predictions, 'r')
##plt.plot(train_dataset[:,-1], 'k')
##plt.show()

# Calculte network accuracy based on test samples
pred = []
for row in test_dataset:
    pred.append(net.predict(row))
pred = np.array(pred).flatten()
yt = test_dataset[:,-1]
test_error = [(a-b)**2 for a, b in zip(pred, yt)]
test_error = np.array(test_error).flatten()
rmse = sum(test_error)/len(test_error)
r2 = 1 - sum([(t-p)**2 for t,p in zip(yt,pred)])/sum([(t-yt.mean())**2 for t in yt])
mae = sum([abs(a-p) for a,p in zip(yt, pred)])/len(pred)

print('## Model regression accuracy: R2 = {:.3f}({:.3f})'.format(r2, r2_score(yt,pred)))
print('## RMSE {:.3f}'.format(np.sqrt(rmse)))
print('## MAE {:.3f}'.format(mae))

# plot the test predictions
fig1 = plt.figure('1')
plt.plot(pred, 'r')
plt.plot(yt, 'b')
plt.title("Fuzzy Decision Tree Regression: R2={:.3f}({:.3f}), nrmse={:.6f}".format(r2, r2_score(yt,pred), rmse))
plt.show()
fig2 = plt.figure('2')
plt.scatter(yt, pred)
#plt.show()

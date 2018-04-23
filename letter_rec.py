from mymodel import MyModel
from torch.autograd import Variable
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as functions

# global config
ATTRIBUTE_NUMBER = 16
CLASS_NUMBER = 26
FILE = "letter-recognition.csv"

# load data
data_train = pd.read_csv(FILE, header=None)
data_train.loc[:, 0] = data_train.loc[:, 0].apply(ord)
data_train.loc[:, 0] = data_train.loc[:, 0] - 65
# data_train = data_train.apply(pd.to_numeric)
data_array = data_train.as_matrix()
features = data_array[:, 1:]
labels = data_array[:, 0]
features_tensor = Variable(torch.Tensor(features).float())
labels_tensor = Variable(torch.Tensor(labels).long())

# Hyper parameters
input_neurons = features.shape[1]
hidden_neurons = 13
output_neurons = np.unique(labels).size
learning_rate = 0.01
num_epoch = 1000000

assert input_neurons == ATTRIBUTE_NUMBER
assert output_neurons == CLASS_NUMBER

# NN
net = MyModel(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)
# Cuda
if torch.cuda.is_available():
    features_tensor = features_tensor.cuda()
    labels_tensor = labels_tensor.cuda()
    net.cuda()

all_losses=[]

for epoch in range(num_epoch):
    # Perform forward pass: compute predicted y by passing x to the model.
    pred = net(features_tensor)

    # Compute loss
    loss = loss_func(pred, labels_tensor)
    all_losses.append(loss.data[0])

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(functions.softmax(pred), 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == labels_tensor.cpu().data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss.data[0], 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

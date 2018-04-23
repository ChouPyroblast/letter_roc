import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.nn.functional as functions


class MyModel(torch.nn.Module):
    def __init__(self, number_input, number_hidden, number_output):
        super(MyModel, self).__init__()
        self.hidden = torch.nn.Linear(number_input, number_hidden)
        self.out = torch.nn.Linear(number_hidden, number_output)

    def forward(self, x):
        hidden_input = self.hidden(x)
        hidden_output = functions.sigmoid(hidden_input)
        y_pred = self.out(hidden_output)
        return y_pred
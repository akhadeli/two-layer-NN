import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import add_bias_dim


class TwoLayerSgdOptimizer():
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate


    def update(self, model, X, t):
        '''Given the model, input X, and target t, update the model with
        stochastic gradient descent.
        Returns:
            The loss for this batch.
        '''
        
        # Forward Propagation
        y2 = model.forward(X)
        
        # Back Propagation
        one_hot_t = F.one_hot(t, 10)
        dJ_dy2 = y2 - one_hot_t

        dJ_dy1 = torch.matmul(model.y1.t(), dJ_dy2)
        db_1 = torch.sum(dJ_dy2)

        # Adapted from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        dJ_dW = torch.matmul(dJ_dy2, model.layer2.weight.t()) * model.y1 * (1 - model.y1)

        dJ_dy0 = torch.matmul(X.t(), dJ_dW)
        db_0 = torch.sum(dJ_dW)

        model.layer1.weight = nn.Parameter(model.layer1.weight - self.learning_rate * dJ_dy0)
        model.layer2.weight = nn.Parameter(model.layer2.weight - self.learning_rate * dJ_dy1)

        model.layer1.bias = nn.Parameter(model.layer1.bias - self.learning_rate * db_0)
        model.layer2.bias = nn.Parameter(model.layer2.bias - self.learning_rate * db_1)

        return F.cross_entropy(y2, t)
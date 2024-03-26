import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import add_bias_dim
from tqdm import tqdm
import numpy as np


class TwoLayerClassifier():
    def __init__(self, input_dim=28 * 28, hidden_dim=64, output_dim=10, hidden_activation_function=F.sigmoid, batch_size=100):
        '''Initialize the network.
        Hint: you may use nn.init.xavier_uniform_ to initialize the weights
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_activation_function = hidden_activation_function
        self.batch_size = batch_size

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        self.layer1.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, hidden_dim)))
        self.layer2.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim, output_dim)))

        self.layer1.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.layer2.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, X):
        '''Perform a forward pass and return the outputs
        Args:
            X: input batch (batch_size, 28*28+1)
        Returns:
            predicted probabilities with shape (batch_size, 10)
        '''
        self.z1 = torch.matmul(X, self.layer1.weight) + self.layer1.bias
        self.y1 = self.hidden_activation_function(self.z1)
        self.z2 = torch.matmul(self.y1, self.layer2.weight) + self.layer2.bias
        self.y2 = F.softmax(self.z2, 1)

        return self.y2

    def train(self, optimizer, X_train, t_train, X_val, t_val, max_epoch=50):
        N_train, N_feature = X_train.shape

        best_epoch, best_acc, best_W1, best_W2 = -1, -1, None, None
        train_losses, valid_accs = [], []

        for epoch in tqdm(range(max_epoch)):
            indices = np.random.permutation(N_train)
            X = X_train[indices]
            t = t_train[indices]

            loss_sum = 0
            for batch_start in range(0, N_train, self.batch_size):
                X_batch = X[batch_start:batch_start + self.batch_size]
                t_batch = t[batch_start:batch_start + self.batch_size].reshape(-1)
                X_batch, t_batch = (X_batch), (t_batch)
                loss_sum += optimizer.update(self, X_batch, t_batch)
            loss_avg = loss_sum/N_train
            train_losses.append(loss_avg)

            _, valid_acc, _ = self.predict(X_val, t_val)
            valid_accs.append(valid_acc)

            if valid_acc > best_acc:
                best_epoch, best_acc, best_W1, best_W2 = epoch, valid_acc, self.layer1.weight, self.layer2.weight

        self.layer1.weight = best_W1
        self.layer2.weight = best_W2

        return best_epoch, best_acc, train_losses, valid_accs


    def predict(self, X, t=None):
        acc = None

        with torch.no_grad():
            y = self.forward(X).numpy()

        t_hat = np.argmax(y, axis=1)
        if t is not None:
            t = t.numpy()
            acc = np.mean(t_hat == t.reshape(-1))

        return t_hat, acc, y
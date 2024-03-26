import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 2D convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # 2D dropout
        self.conv2_drop = nn.Dropout2d()
        
        # linear layers
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Reshape
        x = x.view(-1, 320)

        # Continue through network
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
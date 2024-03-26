from utils import readMNISTdata, add_bias_dim
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

import struct
import os

# Created custom dataset for easy indexing through DataLoader
# iterator
class MNISTDataset(Dataset):
    def __init__(self, targets, img_arr, transform=None, target_transform=None):
        self.img_labels = targets
        self.img_dir = img_arr
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_labels.shape[0]
    
    def __getitem__(self, index):
        image = self.img_dir[index]
        label = self.img_labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def readMNISTdata_torch(path):
    # Reshape the data and change type to work with
    # CNN
    with open(os.path.join(path, 't10k-images.idx3-ubyte'), 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open(os.path.join(path, 't10k-labels.idx1-ubyte'), 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open(os.path.join(path, 'train-images.idx3-ubyte'), 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open(os.path.join(path, 'train-labels.idx1-ubyte'), 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    X_test = test_data / 256
    t_test = test_labels

    _X_train = []
    for i in range(X_train.shape[0]):
        _X_train.append([X_train[i][:-1].reshape((28,28))])

    X_train = np.array(_X_train, dtype=np.float32)

    _X_val = []
    _X_test = []
    for i in range(X_val.shape[0]):
        _X_val.append([X_val[i][:-1].reshape((28,28))])
        _X_test.append([X_test[i][:-1].reshape((28,28))])

    # Ensures that the final X_* can enter the network

    X_val = np.array(_X_val, dtype=np.float32)
    X_test = np.array(_X_test, dtype=np.float32)

    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)
    t_train = torch.from_numpy(t_train.flatten())
    t_val = torch.from_numpy(t_val.flatten())
    t_test = torch.from_numpy(t_test.flatten())

    return X_train, t_train, X_val, t_val, X_test, t_test

def _train(cnn, t_loader, optimizer, train_losses):
    cnn.train()

    loss_sum = 0

    for i, (images, target) in enumerate(t_loader):
        # initialize gradients to zero
        optimizer.zero_grad()

        # start forward pass           
        output = cnn(images)

        # using cross entropy loss
        loss = F.cross_entropy(output, target)
        
        # back propagation steps
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    # Accumulating loss per epoch    
    train_losses.append(loss_sum/len(t_loader.dataset))
    return train_losses

def test(cnn, t_loader, accs):
    # Test the model
    cnn.eval()
    test_loss = 0
    correct = 0

    # safety
    with torch.no_grad():
        for images, labels in t_loader:
            output = cnn(images)
            test_loss += F.cross_entropy(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
        test_loss /= len(t_loader.dataset)
        accs.append(np.float32(correct/len(t_loader.dataset)))
    
    return accs, test_loss
        

def train(num_epochs, cnn, loaders, learning_rate):
    train_losses = []
    accs = []

    # Steepest Gradient Descent
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)
        
    for epoch in tqdm(range(num_epochs)):
        # train
        train_losses = _train(cnn, loaders['train'], optimizer, train_losses)
        
        # validation
        accs, avgloss = test(cnn, loaders['validation'], accs)
    
    return train_losses, accs
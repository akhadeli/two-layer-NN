# -*- coding: utf-8 -*-
import numpy as np
import struct
import os
import torch


def readMNISTdata(path):
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
    # train_data = np.concatenate(
    #     (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    # test_data = np.concatenate(
    #     (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 255
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 255
    t_val = train_labels[50000:]

    return torch.from_numpy(X_train).float(), torch.from_numpy(t_train).to(torch.int64), torch.from_numpy(X_val).float(), torch.from_numpy(t_val).to(torch.int64), torch.from_numpy(test_data / 255).float(), torch.from_numpy(test_labels).to(torch.int64)




def add_bias_dim(X):
    X = X.view(X.size(0), -1)
    o = torch.ones(X.size(0))[:, None]
    X = torch.cat([X, o], dim=1)
    return X
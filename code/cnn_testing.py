# -*- coding: utf-8 -*-
from cnn_utils import readMNISTdata_torch, MNISTDataset, train, test
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from torch.utils.data import DataLoader

from cnn import CNN

MNIST_PATH = "../MNIST/"


np.set_printoptions(threshold=np.inf)
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata_torch(path=MNIST_PATH)

train_data = MNISTDataset(t_train, X_train)
validation_data = MNISTDataset(t_val, X_val)
test_data = MNISTDataset(t_test, X_test)

loaders = {
    'train': DataLoader(train_data, batch_size=100),
    'validation': DataLoader(validation_data, batch_size=100),
    'test': DataLoader(test_data, batch_size=100)
}

print("Data shape:")
print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

cnn = CNN()
print(cnn)

train_losses, valid_accs = train(50, cnn, loaders, learning_rate=0.01)
best_epoch = np.argmax(valid_accs)
best_acc = valid_accs[best_epoch]

print('Best epoch:', best_epoch)
print('Validation acc:', best_acc)

sns.lineplot(x=range(len(train_losses)), y=train_losses)
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epoch")
plt.title("Training Loss")
plt.savefig('cnn_train_loss.png')
plt.clf()
sns.lineplot(x=range(len(valid_accs)), y=valid_accs)
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Validation Accuracy")
plt.savefig('cnn_valid_acc.png')
plt.clf()

accs, avgloss = test(cnn, loaders['test'], [])
print('test acc:', accs[0])
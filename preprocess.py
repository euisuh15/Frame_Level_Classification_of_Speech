import numpy as np
import torch

context = 30

train = np.load('train.npy', allow_pickle=True)
train = np.vstack(train)
train = np.pad(train, ((context, context), (0, 0)), 'constant', constant_values=0)
torch.save(torch.as_tensor(train), 'train_with_pad30.pth')
del train

dev = np.load('dev.npy', allow_pickle=True)
dev = np.vstack(dev)
dev = np.pad(dev, ((context, context), (0, 0)), 'constant', constant_values=0)
torch.save(torch.as_tensor(dev), 'dev_with_pad30.pth')
del dev

test = np.load('test.npy', allow_pickle=True)
test = np.vstack(test)
test = np.pad(test, ((context, context), (0, 0)), 'constant', constant_values=0)
torch.save(torch.as_tensor(test), 'test_with_pad30.pth')
del test


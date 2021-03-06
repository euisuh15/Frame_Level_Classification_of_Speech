import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

import time
import os
from train import *


class MyDataset(data.Dataset):
  def __init__(self, X, Y, context):
    self.X = torch.load(X)

    Y = np.load(Y, allow_pickle=True)
    self.Y = np.hstack(Y)
    self.Y = torch.as_tensor(self.Y)
    self.context = context

  def __len__(self):
    return len(self.Y)

  def __getitem__(self, index):
    low = index+5
    high = index+5+2*self.context

    X = self.X[low:high+1].float().reshape(-1)
    Y = self.Y[index].long()
    return X, Y


# SIMPLE MODEL DEFINITION
class Simple_MLP(nn.Module):
  def __init__(self, size_list):
    super(Simple_MLP, self).__init__()
    layers = []
    self.size_list = size_list

    for i in range(len(size_list)-2):
      layers.append(nn.Linear(size_list[i], size_list[i+1]))
      layers.append(nn.BatchNorm1d(size_list[i+1]))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(p=0.2))

    layers.append(nn.Linear(size_list[-2], size_list[-1]))
    self.net = nn.Sequential(*layers)
  
  def forward(self, x):
    return self.net(x)


cuda = torch.cuda.is_available()
num_workers = 0 if cuda else 0
context = 25

# Training
train_dataset = MyDataset('train_with_pad30.pth', 'train_labels.npy', context)

train_loader_args = dict(shuffle=True, batch_size=512, num_workers=num_workers, pin_memory=True, drop_last=True) if cuda\
                      else dict(shuffle=True, batch_size=64)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Validataion
dev_dataset = MyDataset('dev_with_pad30.pth', 'dev_labels.npy', context)

dev_loader_args = dict(shuffle=False, batch_size=512, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)



model = Simple_MLP([40*(2*context+1), 1024, 1024, 1024, 512, 71])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)


n_epochs = 9
trialNumber = 7

Train_loss = []
Test_loss = []

Train_acc = []
Test_acc = []

# activate when loading model
# model.load_state_dict(torch.load("trial1/model_epoch4.pth"))

for i in range(n_epochs):
  try:
    os.mkdir('trial'+str(trialNumber))
  except:
    pass

  print('Epoch Number: ', i)
  train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
  test_loss, test_acc = val_model(model, train_loader, criterion)

  Train_loss.append(train_loss)
  Test_loss.append(test_loss)
  Train_acc.append(train_acc)
  Test_acc.append(test_acc)

  torch.save(model.state_dict(), 'trial'+str(trialNumber)+'/model_epoch'+str(i)+'.pth')
  print('='*20)
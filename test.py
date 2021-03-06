import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import torch.optim as optim 

class MyDataset(data.Dataset):
  def __init__(self, X, context):
    self.X = np.load(X, allow_pickle=True)
    self.X = np.vstack(self.X)
    self.X = np.pad(self.X, ((context, context), (0, 0)), 'constant', constant_values=0)
    
    self.context = context

    self.length = self.X.shape[0]-2*25

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    low = index
    high = index+2*self.context

    X = self.X[low:high+1].flatten()
    X = torch.as_tensor(X).float()
    return X

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

def Finaltest_model(model, ftest_loader):
  print('Final Testing...')
  with torch.no_grad():
    model.eval()

    p = []
    for batch_idx, (data) in enumerate(ftest_loader):
      
      data = data.to(device)
      outputs = model(data)

      for i in range(len(outputs)):
        t = (torch.argmax(outputs[i])).item()
        p.append(int(t))
      
    return p


cuda = torch.cuda.is_available()

context = 25
num_workers = 0

ftest_dataset = MyDataset('test.npy', context)

ftest_loader_args = dict(shuffle=False, batch_size=512, num_workers=num_workers, pin_memory=True) if cuda\
                      else dict(shuffle=False, batch_size=1)
ftest_loader = data.DataLoader(ftest_dataset, **ftest_loader_args)

model = Simple_MLP([40*(2*context+1), 1024, 1024, 1024, 512, 71])

model.load_state_dict(torch.load('trial3/model_epoch9.pth'))
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)


p = Finaltest_model(model, ftest_loader)

ids = []
for i in range(len(p)):
  ids.append(i)

dp = {'id':ids, 'label':p}

df = pd.DataFrame.from_dict(dp)
df.to_csv('results2.csv', index=False)

print('Done')
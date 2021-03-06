import numpy as np
import torch

def flatten_and_pad(filename, context):
	f = np.load(filename, allow_pickle=True)
	f = np.vstack(f)
	f = np.pad(f, ((context, context), (0, 0)), 'constant', constant_values=0)
	torch.save(torch.as_tensor(f), filename+'_with_pad'+str(context)+'.pth')
	del f

context = 30

flatten_and_pad('train.npy', context)
flatten_and_pad('dev.npy', context)
flatten_and_pad('test.npy', context)


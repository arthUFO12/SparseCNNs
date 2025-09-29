import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from torch import Tensor

from sparse_network import SparseNetwork

image_size = 28
input_size = image_size * image_size
batch_size = 32

net = SparseNetwork(input_size, 256, 128, 10, torch.optim.SGD, 0.001)

net.masking(0.7)
net.apply_masks()


trainset = datasets.MNIST(root='../../datasets',train=True, transform=transforms.ToTensor(), download=False)
testset = datasets.MNIST(root='../../datasets', train=False, transform=transforms.ToTensor(), download=False)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

net.train_model(300, trainloader)
torch.save(net.state_dict(), '../models/sparsity-70.pth')
net.eval_model(testloader)
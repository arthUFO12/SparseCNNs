import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from torch import Tensor

from sparse_convolutional import SparseConvolutionalNetwork

batch_size = 32

net = SparseConvolutionalNetwork(32, 64, 128, 128, 10, torch.optim.SGD, 0.001, 'cuda')

net.masking(0.5)
net.apply_masks()

trainset = datasets.CIFAR10(root='../../datasets',train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.CIFAR10(root='../../datasets', train=False, transform=transforms.ToTensor(), download=True)

trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)


net.train_model(800, trainloader)
torch.save(net.state_dict(), '../models/conv-sparsity-70.pth')
net.eval_model(testloader)
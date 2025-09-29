import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import DataLoader
from torch import Tensor


def apply_hook(module: nn.Linear, mask: Tensor):
    def hook(_, __):
        module.weight.data *= mask
    return hook

class SparseConvolutionalNetwork(nn.Module):
    def __init__(self, out1: int = 32, out2: int = 64, out3: int = 128, fc1out: int = 128, classes: int = 10,
                        optimizer=torch.optim.SGD, lr=0.001, device='cuda', log='../logs/conv.out'):
        torch.manual_seed(42)
        super().__init__()

        self.conv1 = nn.Conv2d(3, out1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out1)

        self.conv2 = nn.Conv2d(out1, out2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out2)

        self.conv3 = nn.Conv2d(out2, out3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out3)

        self.pool = nn.MaxPool2d(2, 2) 

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(out3 * 4 * 4, fc1out)
        self.fc2 = nn.Linear(fc1out, classes)

        self.device = device
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.log = open(log, 'w')
        

    def forward(self, X0):
        X1 = self.pool(F.relu(self.bn1(self.conv1(X0))))
        X2 = self.pool(F.relu(self.bn2(self.conv2(X1))))
        X3 = self.pool(F.relu(self.bn3(self.conv3(X2))))

        y0 = X3.view(X3.size(0), -1)
        
        y1 = self.dropout(F.relu(self.fc1(y0)))
        y2 = self.fc2(y1)

        return y2

    def masking(self, sparsity: float):
        self.register_buffer("conv1mask", (torch.rand(self.conv1.weight.shape) > sparsity).float().data)
        self.register_buffer("conv2mask", (torch.rand(self.conv2.weight.shape) > sparsity).float().data)
        self.register_buffer("conv3mask", (torch.rand(self.conv3.weight.shape) > sparsity).float().data)

        self.register_buffer("fc1mask", (torch.rand(self.fc1.weight.shape) > sparsity).float().data)
        self.register_buffer("fc2mask", (torch.rand(self.fc2.weight.shape) > sparsity).float().data)

        self.conv1mask.to(self.device)
        self.conv2mask.to(self.device)
        self.conv3mask.to(self.device)

        self.fc1mask.to(self.device)
        self.fc2mask.to(self.device)


    def apply_masks(self):
        self.conv1.register_forward_pre_hook(apply_hook(self.conv1, self.conv1mask))
        self.conv2.register_forward_pre_hook(apply_hook(self.conv2, self.conv2mask))
        self.conv3.register_forward_pre_hook(apply_hook(self.conv3, self.conv3mask))

        self.fc1.register_forward_pre_hook(apply_hook(self.fc1, self.fc1mask))
        self.fc2.register_forward_pre_hook(apply_hook(self.fc2, self.fc2mask))

    def mask_grads(self):
        self.conv1.weight.grad *= self.conv1mask
        self.conv2.weight.grad *= self.conv2mask
        self.conv3.weight.grad *= self.conv3mask

        self.fc1.weight.grad *= self.fc1mask
        self.fc2.weight.grad *= self.fc2mask

    def train_model(self, epochs: int, trainloader: DataLoader, lossfunc = nn.CrossEntropyLoss()):
        start = time.time()
        self.train()
        for i in range(epochs):
            tot_corr = 0
            for batch_idx, (X, y) in enumerate(trainloader, start=1):

                y_hat = self(X)

                loss = lossfunc(y_hat, y)

                predicted = torch.max(y_hat.data, 1)[1]
                batch_corr = (predicted == y).sum()
                tot_corr += batch_corr

                self.optimizer.zero_grad()
                loss.backward()
                self.mask_grads()
                self.optimizer.step()

                if batch_idx % 600 == 0:
                    print(f'Epoch: {i} Batch: {batch_idx} Loss: {loss.item()}')
                    print(f'Accuracy: {tot_corr / (batch_idx * trainloader.batch_size)}')

        end = time.time()
        print(f'Training took {end - start} seconds.')
    
    @torch.no_grad()
    def eval_model(self, testloader: DataLoader):
        self.eval()
        tot_corr = 0
        tot = 0
        for idx, (X, y) in enumerate(testloader):
            y_pred = self(X)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y).sum()
            tot_corr += batch_corr
            tot += torch.numel(y)
        
        print(f'Test accuracy: {tot_corr / tot}')
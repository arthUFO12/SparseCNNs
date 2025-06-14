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

class SparseNetwork(nn.Module):
    def __init__(self, input_size: int, hidden1_size: int, hidden2_size: int, output_size:int, optimizer, lr:float, device='cuda'):
        torch.manual_seed(41)
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size).to(device)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size).to(device)
        self.layer3 = nn.Linear(hidden2_size, output_size).to(device)
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.device = torch.device(device)

    def forward(self, x: Tensor):
        x0 = F.relu(self.layer1(x.view(x.size(0), -1)))
        x1 = F.relu(self.layer2(x0))
        return self.layer3(x1)
    
    def masking(self, sparsity: float):
        self.mask1 = (torch.rand(self.layer1.weight.shape) > sparsity).float().data.to(self.device)
        self.mask2 = (torch.rand(self.layer2.weight.shape) > sparsity).float().data.to(self.device)
        self.mask3 = (torch.rand(self.layer3.weight.shape) > sparsity).float().data.to(self.device)

    def apply_masks(self):
        self.layer1.register_forward_pre_hook(apply_hook(self.layer1, self.mask1))
        self.layer2.register_forward_pre_hook(apply_hook(self.layer2, self.mask2))
        self.layer3.register_forward_pre_hook(apply_hook(self.layer3, self.mask3))

    def mask_grads(self):
        self.layer1.weight.grad *= self.mask1
        self.layer2.weight.grad *= self.mask2
        self.layer3.weight.grad *= self.mask3

    def train_model(self, epochs: int, trainloader: DataLoader, lossfunc=torch.nn.CrossEntropyLoss()):
        start = time.time()
        for i in range(epochs):
            tot_corr = 0
            for batch_idx, (X, y) in enumerate(trainloader):
                X, y = X.to(self.device), y.to(self.device)
                batch_idx += 1
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
        tot_corr = 0
        tot = 0
        for idx, (X, y) in enumerate(testloader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self(X)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y).sum()
            tot_corr += batch_corr
            tot += torch.numel(y)
        
        print(f'Test accuracy: {tot_corr / tot}')


    
            

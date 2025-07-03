import torch
import torch.nn as nn
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from sparse_network import SparseNetwork
from argparse import ArgumentParser
from torch.utils.data import DataLoader

# Change this to fit your own directory
dataset_folder = '../../datasets'

models = {'SparseNetwork': SparseNetwork}
datas = {'MNIST': datasets.MNIST,
         'CIFAR10': datasets.CIFAR10,
         'CIFAR100': datasets.CIFAR100 }


@torch.no_grad()
def eval(model: nn.Module,  testloader: DataLoader):
    tot_corr = 0
    tot = 0
    for idx, (X, y) in enumerate(testloader):
        X, y = X.to(model.device), y.to(model.device)
        y_pred = model(X)
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y).sum()
        tot_corr += batch_corr
        tot += torch.numel(y)
    
    print(f'Test accuracy: {tot_corr / tot}')

def visualize_layers(model: nn.Module, folder_path: str):
    plt.axis('off')
    torch.set_printoptions(threshold=float('inf'))
    for param in model.modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            img = np.abs(np.array(param.weight.data.cpu()))
            plt.imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
            plt.savefig(f"{folder_path}/{param}.png")

def compute_sparsity(model: nn.Module):
    for param in model.modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            print((param.weight == 0).sum() / param.weight.numel())
        
def parseArgs():
    args = ArgumentParser(description="Utility program.")
    args.add_argument("--model", help="Type of model being used.")
    args.add_argument("--path", help="Path to the model being used")
    args.add_argument("--architecture", nargs='+', help="Model architecture dimensions", required=True)
    args.add_argument("--action", help="Function to call")
    args.add_argument("--dataset", help="Dataset to use for evaluation", required=False)
    args.add_argument("--plotdir", help="Path to plot directory", required=False)

    return args.parse_args()

def main():
    args = parseArgs()
    args.architecture = [int(s) for s in args.architecture]
    model = models[args.model](*args.architecture)
    model.load_state_dict(torch.load(args.path))

    if args.action == "visualize":
        visualize_layers(model, args.plotdir)
    
    if args.action == "eval":
        test_data = datas[args.dataset](root=dataset_folder, train=False, transform=transforms.ToTensor(), download=False)
        dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
        eval(model, dataloader)


if __name__ == '__main__':
    main()


import torch
import torch.nn as nn
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from sparse_network import SparseNetwork
from sparse_convolutional import SparseConvolutionalNetwork
from argparse import ArgumentParser
from torch.utils.data import DataLoader

# Change this to fit your own directory
dataset_folder = '../../datasets'

models = {'SparseNetwork': SparseNetwork,
          'SparseConvolutionalNetwork': SparseConvolutionalNetwork}

datas = {'MNIST': datasets.MNIST,
         'CIFAR10': datasets.CIFAR10,
         'CIFAR100': datasets.CIFAR100}


def eval(model,  testloader: DataLoader):
    model.eval_model(testloader)

def visualize_layers(model: nn.Module, folder_path: str):
    plt.axis('off')
    torch.set_printoptions(threshold=float('inf'))
    for param in model.modules():
        if isinstance(param, nn.Linear):
            img = np.abs(np.array(param.weight.data.cpu()))
            plt.imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
            plt.savefig(f"{folder_path}/{param}.png")
        elif isinstance(param, nn.Conv2d):
            out_channels, in_channels, kh, kw = param.weight.data.shape
            img = np.ones((in_channels * (kh + 2) + 2, out_channels * (kw + 2) + 2))

            for i in range(in_channels):
                for j in range(out_channels):
                    # Extract kernel and convert to numpy
                    kernel = param.weight.data[j, i, :, :].cpu().numpy()

                    # Compute placement in the grid
                    row_start = i * (kh + 2) + 2
                    row_end   = row_start + kh
                    col_start = j * (kw + 2) + 2
                    col_end   = col_start + kw

                    # Place kernel into grid
                    img[row_start:row_end, col_start:col_end] = kernel

            img = np.abs(img)
            plt.imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
            plt.savefig(f"{folder_path}/{param}.png", dpi=600)

def compute_sparsity(model: nn.Module):
    print((model.conv2mask == 0).sum() / model.conv2mask.numel())
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
    model.masking(0)
    model.load_state_dict(torch.load(args.path))

    if args.action == "visualize":
        visualize_layers(model, args.plotdir)
    
    elif args.action == "eval":
        test_data = datas[args.dataset](root=dataset_folder, train=False, transform=transforms.ToTensor(), download=False)
        dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
        eval(model, dataloader)
    
    elif args.action == "sparsity":
        compute_sparsity(model)


if __name__ == '__main__':
    main()


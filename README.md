# SparseCNNs
## Arthur Ufongene, 2025
Experimentation with sparse implementations of convolutional and linear neural networks.

This repository experiments with different implementations of sparsity in neural networks, such as static, dynamic, and post training sparsity.

## Static Masking
I created 2 computer vision models using static masking. A simple 1-channel model used on the MNIST datset and a 3-channel CNN used on the CIFAR10 dataset. For each, sparsities were set to 0.8. Upon initialization, a random selection of parameters were zeroed out remained at zero for the duration of training.

### Results
MNIST: 96% test accuracy. Expected since the MNIST dataset is pretty easy to train on. Pushing sparsity to 0.9 decreased accuracy a bit, but still achieved 90%.
CIFAR: 74% test accuracy. This wasn't expected, but understandable because I used a simple CNN architecture that probably wasn't the best suited for encoding sparse information. Ranging the number of epochs from 300-800 helped little. In the future, I may try a more suitable architecture or test at what sparsity performance becomes acceptable.

## *Upcoming* RigL Style Masking
RigL style masking employs dynamic masks (masks that change throughout the training process) to increase performance and find the best topology for the model. The parameters with the smallest magnitude gradients are pruned, and the largest are grown if the were previously pruned. It will be interesting to compare the performance of this model against the poor performing convolutional static mask model.

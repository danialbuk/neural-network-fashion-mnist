# PyTorch Fashion MNIST Classification

This repository offers a PyTorch image classification project. It implements and trains a simple feedforward neural network on the FashionMNIST dataset, showcasing data loading, model definition, training loops, and evaluation. The code demonstrates fundamental PyTorch deep learning concepts and includes device-agnostic support for GPU utilization.

## Features

* **Neural Network Implementation:** Defines a simple feedforward neural network using `torch.nn.Module`.
* **FashionMNIST Dataset:** Utilizes the `torchvision.datasets.FashionMNIST` for training and testing.
* **Data Loading:** Employs `torch.utils.data.DataLoader` for efficient batch processing.
* **Training Loop:** Includes a function to train the model with backpropagation and an optimizer (`SGD`).
* **Evaluation:** Provides a function to evaluate the model's accuracy and loss on the test set.
* **Device Agnostic:** Automatically uses a GPU (CUDA) if available, otherwise falls back to CPU.

## Requirements

* Python 3.x
* PyTorch
* Torchvision

You can install the necessary libraries using pip:
```bash
pip install torch torchvision
```

## Usage

To run this project, simply execute the Python script:

```bash
python pytorch_proj_1.py
```

The script will download the FashionMNIST dataset (if not already present), define and train the neural network, and then print the training loss and test accuracy for each epoch.
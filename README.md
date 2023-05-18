---
# Pytorch-Digit-Detector

This repository contains Python code for a simple convolutional neural network (CNN) model trained on the MNIST dataset using PyTorch.

## Installation

Firstly, make sure that Python and pip (Python's package installer) are both installed on your system. You can download Python from its [official site](https://www.python.org/) and pip is included in Python 3.4 and later versions.

To install the required libraries, use pip:

```
pip install pillow torch torchvision
```

Or if you are using a Jupyter notebook, prefix it with an exclamation mark:

```
!pip install pillow torch torchvision
```

## Usage

The script downloads the MNIST dataset, trains a CNN model on it, saves the model as 'model.pt', and then uses this model to classify images 'img_1.jpg' and 'img_5.jpg'.

To run the script, use the following command:

```
python image_classifier.py
```

The script checks if a trained model already exists before initiating the training. If 'model.pt' exists, the script will skip the training phase and directly load the model from the file.

## Model Architecture

The model is a simple CNN with the following architecture:

1. Convolutional layer with 32 filters of size 3x3.
2. ReLU activation.
3. Convolutional layer with 64 filters of size 3x3.
4. ReLU activation.
5. Convolutional layer with 64 filters of size 3x3.
6. ReLU activation.
7. Flatten layer to transform the 3D output to 1D.
8. Linear layer with 30 output nodes.

The Adam optimizer is used with a learning rate of 1e-3. The loss function is CrossEntropyLoss.

## License

This project is licensed under the terms of the MIT license.

## Contact
[Alexandre Hachey](https://www.linkedin.com/in/alexandre-hachey-095210254/) - alx99102@gmail.com

[![linkedin][linkedin-shield]][linkedin-url] 

Project Link: [github.com/alx99102/Pytorch-Digit-Detector](https://github.com/alx99102/Pytorch-Digit-Detector/)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/in/alexandre-hachey-095210254/

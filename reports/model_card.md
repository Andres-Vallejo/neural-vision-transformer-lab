# Model Card: Neural Vision Transformer Lab

## Intended Use

Educational and portfolio demonstration of image classification and representation learning workflows.

## Dataset

Default pipeline uses CIFAR10 via torchvision. The same loader supports ImageFolder datasets for custom image classification.

## Architectures

- CNNClassifier for convolutional baseline performance.
- ConvAutoencoder for unsupervised representation learning.
- ViTClassifier for Transformer encoder based image classification.

## Evaluation

The evaluation script exports classification report and confusion matrix CSV files. Autoencoder quality should be evaluated with reconstruction loss and visual inspection.

## Responsible Use

This repo is not intended for safety-critical image decisions. Any production use would require larger datasets, bias checks, monitoring, and domain-specific validation.

## Suggested Experiments

- Compare CNN and ViT accuracy over 5, 10, and 25 epochs.
- Train the autoencoder and reuse its encoder as a frozen feature extractor.
- Replace CIFAR10 with a domain image dataset such as defects, plants, or medical images.

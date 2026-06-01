# Neural Vision Transformer Lab

Deep learning portfolio project for image classification and representation learning. The project compares three neural approaches on image data:

1. A baseline Convolutional Neural Network classifier.
2. A Convolutional Autoencoder used as an encoder for compact image representations.
3. A Vision Transformer style classifier built with patch embeddings and Transformer encoder blocks.

## Why This Project Matters

Image analytics is one of the strongest ways to demonstrate deep learning skill because it shows data pipelines, GPU training, model architecture, evaluation, and interpretability. This repo is designed to be readable by recruiters while still showing real engineering depth.

## Models Included

- CNNClassifier: convolutional feature extractor plus dense classification head.
- ConvAutoencoder: encoder-decoder network for representation learning and reconstruction.
- ViTClassifier: patch embedding, positional encoding, class token, TransformerEncoder, and classification head.

## Repository Structure

- configs/default.yaml: training configuration.
- src/data.py: CIFAR10/ImageFolder dataloaders with train, validation, and test splits.
- src/models.py: CNN, autoencoder, and Vision Transformer models.
- src/train.py: training loop for classifiers and autoencoder.
- src/evaluate.py: metrics and confusion matrix export.
- reports/model_card.md: model card and experiment notes.

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py --model cnn --epochs 3
python src/train.py --model autoencoder --epochs 3
python src/train.py --model vit --epochs 3
python src/evaluate.py --model vit
```

By default the code downloads CIFAR10 through torchvision, so it works without manually collecting images. The official training set is split into train/validation and evaluation runs on the official test split. You can also point the loader to an ImageFolder dataset.

## Skills Demonstrated

PyTorch, image pipelines, CNNs, autoencoders, encoders, Transformer encoders, training loops, evaluation, model comparison, and model documentation.

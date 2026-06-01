from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_transforms(image_size: int = 32, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def get_dataloaders(data_dir: str, dataset_name: str = "cifar10", image_size: int = 32, batch_size: int = 64, num_workers: int = 2, image_folder: Optional[str] = None):
    root = Path(data_dir)
    if image_folder:
        dataset = datasets.ImageFolder(image_folder, transform=build_transforms(image_size, train=True))
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        class_names = dataset.classes
    elif dataset_name.lower() == "cifar10":
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=build_transforms(image_size, train=True))
        val_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=build_transforms(image_size, train=False))
        class_names = CIFAR10_CLASSES
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_names

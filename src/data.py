from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
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


def get_dataloaders(data_dir: str, name: Optional[str] = None, dataset_name: str = "cifar10", image_size: int = 32, batch_size: int = 64, num_workers: int = 2, image_folder: Optional[str] = None, val_split: float = 0.1):
    dataset_name = name or dataset_name
    root = Path(data_dir)
    if image_folder:
        train_dataset = datasets.ImageFolder(image_folder, transform=build_transforms(image_size, train=True))
        eval_dataset = datasets.ImageFolder(image_folder, transform=build_transforms(image_size, train=False))
        val_size = max(1, int(val_split * len(train_dataset)))
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42)).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(eval_dataset, val_indices)
        test_ds = val_ds
        class_names = train_dataset.classes
    elif dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=build_transforms(image_size, train=True))
        eval_train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=build_transforms(image_size, train=False))
        val_size = max(1, int(val_split * len(train_dataset)))
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42)).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_ds = Subset(train_dataset, train_indices)
        val_ds = Subset(eval_train_dataset, val_indices)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=build_transforms(image_size, train=False))
        class_names = CIFAR10_CLASSES
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, class_names

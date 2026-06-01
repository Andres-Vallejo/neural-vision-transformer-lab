import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
import yaml
from data import get_dataloaders
from models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(model, loader, criterion, optimizer, device, autoencoder=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        target = images if autoencoder else labels
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        if not autoencoder:
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader.dataset), correct / total if total else None


@torch.no_grad()
def validate(model, loader, criterion, device, autoencoder=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        target = images if autoencoder else labels
        loss = criterion(outputs, target)
        total_loss += loss.item() * images.size(0)
        if not autoencoder:
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader.dataset), correct / total if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["cnn", "autoencoder", "vit"], default="cnn")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = resolve_device(cfg["training"].get("device", "auto"))
    train_loader, val_loader, _, class_names = get_dataloaders(**cfg["dataset"])
    model_cfg = cfg["models"][args.model]
    model = build_model(args.model, model_cfg, num_classes=len(class_names)).to(device)
    autoencoder = args.model == "autoencoder"
    criterion = nn.MSELoss() if autoencoder else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    epochs = args.epochs or cfg["training"]["epochs"]

    ckpt_dir = Path(cfg["outputs"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, autoencoder)
        val_loss, val_acc = validate(model, val_loader, criterion, device, autoencoder)
        print({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4), "train_acc": train_acc, "val_acc": val_acc})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model_state": model.state_dict(), "class_names": class_names, "model": args.model, "config": cfg}, ckpt_dir / f"{args.model}_best.pt")


if __name__ == "__main__":
    main()

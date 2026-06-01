import argparse
from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import yaml
from data import get_dataloaders
from models import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["cnn", "vit"], default="vit")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, class_names = get_dataloaders(**cfg["dataset"])
    model = build_model(args.model, cfg["models"][args.model], num_classes=len(class_names)).to(device)
    checkpoint = torch.load(Path(cfg["outputs"]["checkpoint_dir"]) / f"{args.model}_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    y_true, y_pred = [], []
    for images, labels in test_loader:
        logits = model(images.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics_dir = Path(cfg["outputs"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).T.to_csv(metrics_dir / f"{args.model}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_true, y_pred), index=class_names, columns=class_names).to_csv(metrics_dir / f"{args.model}_confusion_matrix.csv")
    print(pd.DataFrame(report).T)


if __name__ == "__main__":
    main()

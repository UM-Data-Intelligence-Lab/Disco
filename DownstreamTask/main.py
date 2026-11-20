import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from .data import RegionDataset, load_split
from .gdp import build_head, configure_device
from .training import train_model_downstream_moe


PARTITIONS = ("train_ids", "val_ids", "test_ids")


def split_regions(region_ids, labels, seed, classification):
    stratify = [labels[cell] for cell in region_ids] if classification else None
    try:
        remaining, test = train_test_split(region_ids, test_size=0.15, random_state=seed, stratify=stratify)
    except ValueError:
        remaining, test = train_test_split(region_ids, test_size=0.15, random_state=seed)
    stratify = [labels[cell] for cell in remaining] if classification else None
    try:
        train, val = train_test_split(remaining, test_size=0.15 / 0.85, random_state=seed, stratify=stratify)
    except ValueError:
        train, val = train_test_split(remaining, test_size=0.15 / 0.85, random_state=seed)
    return tuple(sorted(subset) for subset in (train, val, test))


def test_metrics(model, loader, device, classification, mean, std, log_target):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            prediction, _ = model(*(batch[name].to(device) for name in ("shared", "functional", "visual")))
            scores.append(prediction.cpu())
            labels.append(batch["label"])

    target = torch.cat(labels).numpy()
    predictions = torch.cat(scores)
    if classification:
        probabilities = predictions.softmax(dim=1).numpy()
        classes = probabilities.argmax(axis=1)
        aucs = [roc_auc_score(target == index, probabilities[:, index])
                for index in np.unique(target) if (target != index).any()]
        return {
            "accuracy": float(accuracy_score(target, classes)),
            "macro_f1": float(f1_score(target, classes, average="macro", zero_division=0)),
            "auc": float(np.mean(aucs)) if aucs else None,
        }

    target = target.astype(np.float64) * std + mean
    predictions = predictions.numpy().reshape(-1).astype(np.float64) * std + mean
    if log_target:
        target = np.expm1(np.clip(target, -30, 30))
        predictions = np.expm1(np.clip(predictions, -30, 30))
    return {
        "r2": float(r2_score(target, predictions)) if len(target) > 1 else None,
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-column", default="target")
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "downstream.json")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    device = configure_device(args.device)
    region_ids = np.load(args.data_dir / "h3_order.npy", allow_pickle=False).tolist()
    embeddings = tuple(np.load(args.data_dir / f"{name}_embeddings.npy", allow_pickle=False)
                       for name in ("shared", "functional", "visual"))
    frame = pd.read_csv(args.labels, dtype=str, usecols=["h3", args.label_column]).dropna()
    frame["h3"] = frame["h3"].str.strip().str.lower()
    labels = frame.set_index("h3")[args.label_column].to_dict()
    labeled_ids = sorted(labels.keys() & set(region_ids))
    labels = {cell: labels[cell] for cell in labeled_ids}
    classification = args.task == "classification"
    if args.split:
        split = load_split(args.split, config["split_seed"], labeled_ids)
    else:
        split = split_regions(labeled_ids, labels, config["split_seed"], classification)

    mean, std = 0.0, 1.0
    if classification:
        vocabulary = {label: index for index, label in enumerate(sorted(set(labels.values())))}
        labels = {key: vocabulary[value] for key, value in labels.items()}
        num_classes = len(vocabulary)
        criterion = nn.CrossEntropyLoss()
        dtype = torch.long
    else:
        values = np.array([float(labels[cell]) for cell in labeled_ids], dtype=np.float32)
        if args.log_target:
            values = np.log1p(values)
        transformed = dict(zip(labeled_ids, values, strict=True))
        train_values = np.array([transformed[cell] for cell in split[0]], dtype=np.float64)
        mean = float(train_values.mean())
        std = float(train_values.std()) + 1e-6
        labels = {cell: float((value - mean) / std) for cell, value in transformed.items()}
        num_classes = 1
        criterion = nn.MSELoss()
        dtype = torch.float32

    train, val, test = (
        DataLoader(RegionDataset(region_ids, embeddings, labels, subset, label_dtype=dtype),
                   batch_size=config["batch_size"], shuffle=(index == 0))
        for index, subset in enumerate(split)
    )
    torch.backends.cudnn.deterministic = True
    rows = []
    for seed in config["seeds"]:
        print(f"Disco (Full) | {args.task} | seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = build_head(embeddings[0].shape[1], config, device, num_classes)
        model = train_model_downstream_moe(model, train, val, criterion, device, config)
        metrics = test_metrics(model, test, device, classification, mean, std, args.log_target)
        rows.append({"seed": seed, **metrics})

    summary = {}
    for metric in metrics:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = {
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = json.dumps({"per_seed": rows, "summary": summary}, indent=2)
    (args.output_dir / "metrics.json").write_text(results + "\n")
    split_data = {"seed": config["split_seed"], **dict(zip(PARTITIONS, split, strict=True))}
    (args.output_dir / "split.json").write_text(json.dumps(split_data, indent=2) + "\n")
    print(results)


if __name__ == "__main__":
    main()

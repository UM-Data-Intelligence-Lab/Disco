import csv
import random
import statistics

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from .data import RegionDataset, normalize_gdp
from .model import DownstreamMoEModel
from .training import train_model_downstream_moe


METHODS = {
    "full": "Disco (Full)",
}
METRICS = ("r2", "mae", "rmse")


def evaluate(model, loader, device, mean, std):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            estimate, _ = model(batch["shared"].to(device),
                                batch["functional"].to(device),
                                batch["visual"].to(device))
            predictions.append(estimate.cpu())
            targets.append(batch["label"])
    prediction = torch.cat(predictions).numpy().squeeze().astype(np.float64)
    target = torch.cat(targets).numpy().squeeze().astype(np.float64)
    p = np.expm1(np.clip(prediction * (std + 1e-6) + mean, -30.0, 30.0)).astype(np.float32)
    y = np.expm1(np.clip(target * (std + 1e-6) + mean, -30.0, 30.0)).astype(np.float32)
    return {
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
    }


def build_head(emb_dim, config, device, num_classes=1):
    return DownstreamMoEModel(
        emb_dim=emb_dim,
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"],
        moe_output_dim=config["moe_output_dim"],
        gate_hidden_dim=config["gate_hidden_dim"],
        dropout=config["dropout"],
    ).to(device)


def configure_device(requested):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def train_gdp(embeddings, region_ids, labels, split, config, device, directory):
    normalized, mean, std = normalize_gdp(labels, split[0])
    directory.mkdir(parents=True, exist_ok=False)
    torch.backends.cudnn.deterministic = True
    for seed in config["seeds"]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Singapore GDP | {METHODS['full']} | seed={seed}", flush=True)
        train, val = (
            DataLoader(RegionDataset(region_ids, embeddings, normalized, subset),
                       batch_size=config["batch_size"], shuffle=(index == 0))
            for index, subset in enumerate(split[:2])
        )
        model = build_head(embeddings[0].shape[1], config, device)
        model = train_model_downstream_moe(model, train, val, nn.MSELoss(), device, config)
        torch.save(model.state_dict(), directory / f"full_{seed}.pt")
    return mean, std


def write_results(directory, rows):
    with (directory / "per_seed.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("method", "seed", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    table = ["| Method | R² ↑ | MAE ↓ | RMSE ↓ |", "|---|---:|---:|---:|"]
    for method in METHODS.values():
        records = [row for row in rows if row["method"] == method]
        cells = []
        for metric in METRICS:
            values = [row[metric] for row in records]
            mean, std = statistics.mean(values), statistics.stdev(values)
            summary.append({"method": method, "metric": metric, "mean": mean, "sample_std": std,
                            "n": len(values)})
            precision = 3 if metric == "r2" else 1
            cells.append(f"{mean:.{precision}f} ({std:.{precision}f})")
        table.append("| " + " | ".join((method, *cells)) + " |")
    with (directory / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("method", "metric", "mean", "sample_std", "n"))
        writer.writeheader()
        writer.writerows(summary)
    markdown = "\n".join(table) + "\n"
    (directory / "sg_gdp.md").write_text(markdown)
    print(markdown)

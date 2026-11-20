import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_gdp(path):
    frame = pd.read_csv(path, dtype=str, usecols=["h3", "gdp"])
    frame["h3"] = frame["h3"].str.strip().str.lower()
    raw = pd.to_numeric(frame["gdp"], errors="raise").to_numpy(dtype=np.float32)
    if not frame["h3"].notna().all() or frame["h3"].duplicated().any():
        raise ValueError("GDP CSV has missing or duplicate H3 identifiers")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("GDP labels must be finite nonnegative values")
    return {key: float(value) for key, value in zip(frame["h3"], np.log1p(raw), strict=True)}


def load_split(path, seed, region_ids):
    split = json.loads(path.read_text())
    if split["seed"] != seed:
        raise ValueError("Split seed does not match config")
    partitions = tuple(split[name] for name in ("train_ids", "val_ids", "test_ids"))
    combined = [key for subset in partitions for key in subset]
    if len(set(combined)) != len(combined) or set(combined) != set(region_ids):
        raise ValueError("Split must partition the GDP regions without duplicate cells")
    return partitions


def normalize_gdp(labels, train_ids):
    train_values = np.asarray([labels[key] for key in train_ids], dtype=np.float64)
    mean = float(train_values.mean())
    std = float(train_values.std(ddof=0))
    if std <= 1e-12:
        raise ValueError("Training GDP labels have no variance")
    normalized = {key: float((value - mean) / (std + 1e-6)) for key, value in labels.items()}
    return normalized, mean, std


class RegionDataset(Dataset):
    def __init__(self, region_ids, embeddings, labels, subset, label_dtype=torch.float32):
        self.ids = sorted(subset)
        self.rows = {key: index for index, key in enumerate(region_ids)}
        self.embeddings = embeddings
        self.labels = labels
        self.label_dtype = label_dtype

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        key = self.ids[index]
        row = self.rows[key]
        names = ("shared", "functional", "visual")
        features = {name: torch.from_numpy(values[row])
                    for name, values in zip(names, self.embeddings, strict=True)}
        features["label"] = torch.tensor(self.labels[key], dtype=self.label_dtype)
        return features

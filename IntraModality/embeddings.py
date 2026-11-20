import numpy as np


MODALITIES = ("poi", "mobility_flow", "streetview", "satellite")


def load_modalities(directory, region_ids, config):
    aligned = {}
    dimension = (*config["functional_input_parts"], *config["visual_input_parts"])
    for name, size in zip(MODALITIES, dimension, strict=True):
        matrix = np.load(directory / f"{name}_embeddings.npy", allow_pickle=False)
        h3_ids = np.load(directory / f"{name}_h3_order.npy", allow_pickle=False)
        if h3_ids.ndim != 1 or matrix.ndim != 2 or matrix.shape != (len(h3_ids), size):
            raise ValueError(f"Invalid embedding or H3 shape for {name}")
        keys = np.char.lower(np.char.strip(h3_ids.astype(str))).tolist()
        if len(set(keys)) != len(keys) or not np.isfinite(matrix).all():
            raise ValueError(f"Duplicate H3 IDs or nonfinite embeddings for {name}")
        by_region = dict(zip(keys, matrix.astype(np.float32, copy=False)))
        absent = set(region_ids) - by_region.keys()
        if absent and name != "streetview":
            raise ValueError(f"{name} is missing embeddings for {len(absent)} GDP regions")
        zero = np.zeros(size, dtype=np.float32)
        aligned[name] = np.stack([by_region.get(h3, zero) for h3 in region_ids])
        print(f"{name}: {len(region_ids)-len(absent)}/{len(region_ids)} regions")
    functional = np.concatenate((aligned["poi"], aligned["mobility_flow"]), axis=1)
    visual = np.concatenate((aligned["streetview"], aligned["satellite"]), axis=1)
    return functional, visual

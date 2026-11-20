import argparse
from pathlib import Path

import numpy as np

from .embeddings import MODALITIES


def align_and_concat(modality_data, modality_names, target_h3):
    aligned = []
    masks = []
    for name in modality_names:
        h3_list, latent = modality_data[name]
        h3_to_idx = {cell: index for index, cell in enumerate(h3_list)}
        rows = np.zeros((len(target_h3), latent.shape[1]), dtype=np.float32)
        mask = np.zeros(len(target_h3), dtype=np.int8)
        for index, cell in enumerate(target_h3):
            if cell in h3_to_idx:
                rows[index] = latent[h3_to_idx[cell]]
                mask[index] = 1
        aligned.append(rows)
        masks.append(mask)
    return np.concatenate(aligned, axis=1), np.stack(masks, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    modality_data = {}
    for name in MODALITIES:
        ids = np.load(args.data_dir / f"{name}_h3_order.npy", allow_pickle=False).astype(str)
        values = np.load(args.data_dir / f"{name}_embeddings.npy", allow_pickle=False)
        modality_data[name] = ([cell.strip().lower() for cell in ids], values)

    functional_names = MODALITIES[:2]
    visual_names = MODALITIES[2:]
    functional_ids = set().union(*(modality_data[name][0] for name in functional_names))
    visual_ids = set().union(*(modality_data[name][0] for name in visual_names))
    region_ids = sorted(functional_ids & visual_ids)
    functional, functional_mask = align_and_concat(modality_data, functional_names, region_ids)
    visual, visual_mask = align_and_concat(modality_data, visual_names, region_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "functional_input.npy", functional)
    np.save(args.output_dir / "visual_input.npy", visual)
    np.save(args.output_dir / "h3_order.npy", np.asarray(region_ids))
    np.save(args.output_dir / "modality_mask.npy", np.concatenate((functional_mask, visual_mask), axis=1))


if __name__ == "__main__":
    main()

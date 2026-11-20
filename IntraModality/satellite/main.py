import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


SAT_DIR = os.path.dirname(os.path.abspath(__file__))

from ..encoder import ImageEncoder
from ..decoder import LatentPredictor
from ..contrastive import ContrastiveModule
from ..reconstruction import ReconstructionModule
from .satellite_feature import SatelliteFeatureLoader
from .augment import SatelliteAugmentor


DATA_PATH = None
SAVE_DIR = None

LATENT_DIM = 128
PROJ_DIM = 64
TEMPERATURE = 0.07
LAMBDA_CL = 1.0
LAMBDA_RECON = 1.0
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
IMAGE_SIZE = 64
NOISE_SCALES = [0.02, 0.05, 0.10]
BACKBONE = "resnet18"
FREEZE_BACKBONE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CITIES = ["sg"]

def load_satellite_data(city: str) -> dict:
    loader = SatelliteFeatureLoader(DATA_PATH, city, image_size=IMAGE_SIZE)
    img_tensor = loader.get_numpy()
    h3_labels = loader.get_h3_labels()

    augmentor = SatelliteAugmentor(img_tensor)
    aug_data = augmentor.build_all()

    return {
        "original": img_tensor,
        "augmented": aug_data,
        "h3_labels": h3_labels,
    }

def build_satellite_modules() -> dict:
    encoder = ImageEncoder(latent_dim=LATENT_DIM, backbone=BACKBONE, pretrained=True, freeze_backbone=FREEZE_BACKBONE)
    cl_module = ContrastiveModule(encoder=encoder, latent_dim=LATENT_DIM, proj_dim=PROJ_DIM, temperature=TEMPERATURE)

    predictor = LatentPredictor(latent_dim=LATENT_DIM)
    rec_module = ReconstructionModule(encoder=encoder, predictor=predictor, loss_type="cosine")

    return {"encoder": encoder, "predictor": predictor, "contrastive": cl_module, "reconstruction": rec_module}

def train_and_extract(city: str, data: dict, modules: dict):
    print(f"\n{'='*70}")
    print(f"{'='*70}")

    x_orig_np = data["original"]
    aug_dict = data["augmented"]
    N = x_orig_np.shape[0]

    x_orig_t = torch.tensor(x_orig_np, dtype=torch.float32)
    loaders = {}
    for aug_name, mat in aug_dict.items():
        x_aug_t = torch.tensor(mat, dtype=torch.float32)
        dataset = TensorDataset(x_orig_t, x_aug_t)
        loaders[aug_name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=N % BATCH_SIZE == 1)

    cl_module = modules["contrastive"].to(DEVICE)
    rec_module = modules["reconstruction"].to(DEVICE)
    all_modules = nn.ModuleList([cl_module, rec_module])
    optimizer = torch.optim.Adam(all_modules.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    all_modules.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_loss_cl = epoch_loss_recon = 0.0
        num_batches = 0

        for aug_name in loaders.keys():
            for batch_orig, batch_aug in loaders[aug_name]:
                batch_orig, batch_aug = batch_orig.to(DEVICE), batch_aug.to(DEVICE)

                out_cl = cl_module(batch_orig, batch_aug)
                out_rec = rec_module(batch_orig, batch_aug)
                loss = LAMBDA_CL * out_cl["loss_cl"] + LAMBDA_RECON * out_rec["loss_recon"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_cl += out_cl["loss_cl"].item()
                epoch_loss_recon += out_rec["loss_recon"].item()
                num_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | loss={epoch_loss/num_batches:.4f} cl={epoch_loss_cl/num_batches:.4f} recon={epoch_loss_recon/num_batches:.4f}")

    all_modules.eval()
    z_parts = []
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            batch = x_orig_t[i:i+BATCH_SIZE].to(DEVICE)
            z_parts.append(modules["encoder"](batch).cpu())
    final_z = torch.cat(z_parts, dim=0).numpy()


    return final_z, data["h3_labels"]

def save_results(city: str, final_z: np.ndarray, final_h3_list: list, modules: dict):
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.save(os.path.join(SAVE_DIR, "satellite_embeddings.npy"), final_z)
    np.save(os.path.join(SAVE_DIR, "satellite_h3_order.npy"), np.array(final_h3_list))

    torch.save({
        "encoder": modules["encoder"].state_dict(),
        "predictor": modules["predictor"].state_dict(),
        "contrastive": modules["contrastive"].state_dict(),
        "reconstruction": modules["reconstruction"].state_dict(),
    }, os.path.join(SAVE_DIR, "satellite_model.pt"))

def main():
    global DATA_PATH, SAVE_DIR, EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city", default=CITIES[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    EPOCHS = args.epochs
    DATA_PATH = str(args.data_dir.resolve())
    SAVE_DIR = str(args.output_dir.resolve())
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("=" * 70)

    for city in (args.city,):
        data = load_satellite_data(city)
        modules = build_satellite_modules()
        final_z, final_h3_list = train_and_extract(city, data, modules)
        save_results(city, final_z, final_h3_list, modules)

if __name__ == "__main__":
    main()

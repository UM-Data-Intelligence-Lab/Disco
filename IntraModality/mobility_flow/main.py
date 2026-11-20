import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


FLOW_DIR = os.path.dirname(os.path.abspath(__file__))

from ..encoder import MLPEncoder
from ..decoder import LatentPredictor
from ..contrastive import ContrastiveModule
from ..reconstruction import ReconstructionModule
from .flow_feature import FlowFeatureLoader
from .augment import FlowAugmentor


DATA_PATH = None

SAVE_DIR = None

LATENT_DIM = 128
HIDDEN_DIM = 256
PROJ_DIM = 64
DROPOUT = 0.1
TEMPERATURE = 0.07
LAMBDA_CL = 1.0
LAMBDA_RECON = 1.0
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
NOISE_SCALES = [0.05, 0.10, 0.20]
NORM = "log1p_zscore"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CITIES = ["singapore"]


def load_flow_data(city: str) -> dict:
    loader = FlowFeatureLoader(DATA_PATH, city, norm=NORM)
    flow_matrix = loader.get_numpy()
    h3_list = loader.get_h3_list()

    N, D = flow_matrix.shape

    augmentor = FlowAugmentor(flow_matrix, noise_scales=NOISE_SCALES)
    aug_data = augmentor.build_all()

    return {
        "original": flow_matrix,
        "augmented": aug_data,
        "h3_list": h3_list,
    }

def build_flow_modules(input_dim: int) -> dict:
    encoder = MLPEncoder(input_dim=input_dim, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    cl_module = ContrastiveModule(encoder=encoder, latent_dim=LATENT_DIM, proj_dim=PROJ_DIM, temperature=TEMPERATURE)

    predictor = LatentPredictor(latent_dim=LATENT_DIM)
    rec_module = ReconstructionModule(encoder=encoder, predictor=predictor, loss_type="cosine")

    return {"encoder": encoder, "predictor": predictor, "contrastive": cl_module, "reconstruction": rec_module}

def train_city(city: str, data: dict, modules: dict) -> torch.Tensor:
    print(f"\n{'='*70}")
    print(f"{'='*70}")

    x_orig_np = data["original"]
    aug_dict = data["augmented"]
    N = x_orig_np.shape[0]

    x_orig_t = torch.tensor(x_orig_np, dtype=torch.float32)
    aug_tensors = {name: torch.tensor(mat, dtype=torch.float32) for name, mat in aug_dict.items()}

    loaders = {}
    for aug_name, x_aug_t in aug_tensors.items():
        dataset = TensorDataset(x_orig_t, x_aug_t)
        loaders[aug_name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=N % BATCH_SIZE == 1)

    cl_module = modules["contrastive"].to(DEVICE)
    rec_module = modules["reconstruction"].to(DEVICE)
    all_modules = nn.ModuleList([cl_module, rec_module])
    optimizer = torch.optim.Adam(all_modules.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    aug_names = list(loaders.keys())

    all_modules.train()
    for epoch in range(EPOCHS):
        epoch_loss = epoch_loss_cl = epoch_loss_recon = 0.0
        num_batches = 0

        for aug_name in aug_names:
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
            avg_loss, avg_cl, avg_recon = epoch_loss / num_batches, epoch_loss_cl / num_batches, epoch_loss_recon / num_batches
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | loss={avg_loss:.4f}  cl={avg_cl:.4f}  recon={avg_recon:.4f}")

    all_modules.eval()
    with torch.no_grad():
        z_final = modules["encoder"].to(DEVICE)(x_orig_t.to(DEVICE))

    return z_final

def save_results(city: str, z: torch.Tensor, data: dict, modules: dict):
    os.makedirs(SAVE_DIR, exist_ok=True)

    z_np = z.cpu().numpy()
    print(f"   Embedding mean={z_np.mean():.4f}  std={z_np.std():.4f}")
    np.save(os.path.join(SAVE_DIR, "mobility_flow_embeddings.npy"), z_np)
    np.save(os.path.join(SAVE_DIR, "mobility_flow_h3_order.npy"), np.array(data["h3_list"]))

    ckpt = {
        "encoder": modules["encoder"].state_dict(),
        "predictor": modules["predictor"].state_dict(),
        "contrastive": modules["contrastive"].state_dict(),
        "reconstruction": modules["reconstruction"].state_dict(),
    }
    torch.save(ckpt, os.path.join(SAVE_DIR, "flow_model.pt"))

@torch.no_grad()
def evaluate(city: str, data: dict, modules: dict):
    x_orig = torch.tensor(data["original"], dtype=torch.float32).to(DEVICE)
    modules["contrastive"].eval()
    modules["reconstruction"].eval()


    for aug_name, aug_np in data["augmented"].items():
        x_aug = torch.tensor(aug_np, dtype=torch.float32).to(DEVICE)
        out_cl = modules["contrastive"](x_orig, x_aug)
        out_rec = modules["reconstruction"](x_orig, x_aug)
        print(f"   [{aug_name:10s}]  CL loss={out_cl['loss_cl']:.4f}  Latent Recon loss={out_rec['loss_recon']:.4f}")

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
        data = load_flow_data(city)
        modules = build_flow_modules(input_dim=data["original"].shape[1])
        z = train_city(city, data, modules)
        evaluate(city, data, modules)
        save_results(city, z, data, modules)


if __name__ == "__main__":
    main()

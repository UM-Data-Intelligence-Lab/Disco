import json
import argparse
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler


SV_DIR = os.path.dirname(os.path.abspath(__file__))

from ..encoder import ImageEncoder
from ..decoder import LatentPredictor
from ..contrastive import ContrastiveModule
from ..reconstruction import ReconstructionModule
from .streetview_feature import StreetviewFeatureLoader


DATA_PATH = None

SAVE_DIR = None
CACHE_DIR = None

LATENT_DIM = 128
PROJ_DIM = 64
TEMPERATURE = 0.07
LAMBDA_CL = 1.0
LAMBDA_RECON = 1.0
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
IMAGE_SIZE = 64
NOISE_SCALES = [0.02, 0.05, 0.10]
BACKBONE = "resnet18"
FREEZE_BACKBONE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CITIES = ["singapore"]


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_streetview_data(city: str) -> dict:
    cache_img = os.path.join(CACHE_DIR, f"{city}_images.npy")
    cache_labels = os.path.join(CACHE_DIR, f"{city}_h3_labels.json")

    if os.path.exists(cache_img) and os.path.exists(cache_labels):
        img_tensor = np.load(cache_img, mmap_mode='r')
        img_tensor = np.array(img_tensor)
        with open(cache_labels, 'r') as f:
            h3_labels = json.load(f)
        return {"original": img_tensor, "h3_labels": h3_labels}

    loader = StreetviewFeatureLoader(DATA_PATH, city, image_size=IMAGE_SIZE)
    img_tensor = loader.image_tensor
    h3_labels = loader.h3_labels

    np.save(cache_img, img_tensor)
    with open(cache_labels, 'w') as f:
        json.dump(h3_labels, f)

    return {"original": img_tensor, "h3_labels": h3_labels}


def build_streetview_modules() -> dict:
    encoder = ImageEncoder(
        latent_dim=LATENT_DIM, backbone=BACKBONE,
        pretrained=True, freeze_backbone=FREEZE_BACKBONE
    )
    cl_module = ContrastiveModule(
        encoder=encoder, latent_dim=LATENT_DIM,
        proj_dim=PROJ_DIM, temperature=TEMPERATURE
    )
    predictor = LatentPredictor(latent_dim=LATENT_DIM)
    rec_module = ReconstructionModule(
        encoder=encoder, predictor=predictor, loss_type="cosine"
    )
    return {
        "encoder": encoder, "predictor": predictor,
        "contrastive": cl_module, "reconstruction": rec_module,
    }


def train_and_aggregate(city: str, data: dict, modules: dict):
    print(f"\n{'='*70}")
    print(f"{'='*70}")

    M = data["original"].shape[0]
    train_size = M - (M % BATCH_SIZE == 1)
    data_size_bytes = data["original"].nbytes
    gpu_resident = False
    if DEVICE == "cuda":
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        gpu_used = torch.cuda.memory_allocated()
        gpu_free = gpu_total - gpu_used
        gpu_resident = data_size_bytes < gpu_free * 0.35

    x_cpu = torch.from_numpy(data["original"]).float()

    if gpu_resident:
        x_gpu = x_cpu.to(DEVICE).to(memory_format=torch.channels_last)
        del x_cpu
    else:
        x_gpu = None


    cl_module = modules["contrastive"].to(DEVICE).to(memory_format=torch.channels_last)
    rec_module = modules["reconstruction"].to(DEVICE).to(memory_format=torch.channels_last)

    all_params = list(set(list(cl_module.parameters()) + list(rec_module.parameters())))
    optimizer = torch.optim.Adam(all_params, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    use_amp = (DEVICE == "cuda")

    scaler = GradScaler('cuda', enabled=use_amp)

    print(f"   M={M}, Batch={BATCH_SIZE}, Epochs={EPOCHS}, AMP={use_amp}")

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    cl_module.train()
    rec_module.train()

    for epoch in range(EPOCHS):
        t_epoch = time.time()
        epoch_loss = epoch_loss_cl = epoch_loss_recon = 0.0
        num_batches = 0

        if gpu_resident:

            perm = torch.randperm(M, device=DEVICE)
            for start in range(0, train_size, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                batch_orig = x_gpu[idx]

                scale = NOISE_SCALES[torch.randint(0, len(NOISE_SCALES), (1,)).item()]
                batch_aug = (batch_orig + torch.randn_like(batch_orig) * scale).clamp_(0, 1)


                with torch.amp.autocast('cuda', enabled=use_amp):
                    out_cl = cl_module(batch_orig, batch_aug)
                    out_rec = rec_module(batch_orig, batch_aug)
                    loss = LAMBDA_CL * out_cl["loss_cl"] + LAMBDA_RECON * out_rec["loss_recon"]

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_loss_cl += out_cl["loss_cl"].item()
                epoch_loss_recon += out_rec["loss_recon"].item()
                num_batches += 1
        else:

            perm = torch.randperm(M)
            for start in range(0, train_size, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                batch_orig = x_cpu[idx].to(DEVICE, non_blocking=True)
                batch_orig = batch_orig.to(memory_format=torch.channels_last)

                scale = NOISE_SCALES[torch.randint(0, len(NOISE_SCALES), (1,)).item()]
                batch_aug = (batch_orig + torch.randn_like(batch_orig) * scale).clamp_(0, 1)


                with torch.amp.autocast('cuda', enabled=use_amp):
                    out_cl = cl_module(batch_orig, batch_aug)
                    out_rec = rec_module(batch_orig, batch_aug)
                    loss = LAMBDA_CL * out_cl["loss_cl"] + LAMBDA_RECON * out_rec["loss_recon"]

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                epoch_loss_cl += out_cl["loss_cl"].item()
                epoch_loss_recon += out_rec["loss_recon"].item()
                num_batches += 1

                del batch_orig, batch_aug

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t_epoch
            n = max(num_batches, 1)
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            print(f"   Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"loss={epoch_loss/n:.4f} "
                  f"cl={epoch_loss_cl/n:.4f} "
                  f"recon={epoch_loss_recon/n:.4f} | "
                  f"{dt:.1f}s | mem={mem_gb:.1f}GB")


    encoder = modules["encoder"].to(DEVICE).to(memory_format=torch.channels_last)
    encoder.eval()

    INFER_BATCH = BATCH_SIZE * 4
    z_parts = []
    with torch.no_grad():

        with torch.amp.autocast('cuda', enabled=use_amp):
            for i in range(0, M, INFER_BATCH):
                if gpu_resident:
                    batch = x_gpu[i:i + INFER_BATCH]
                else:
                    batch = x_cpu[i:i + INFER_BATCH].to(DEVICE, non_blocking=True)
                    batch = batch.to(memory_format=torch.channels_last)
                z_parts.append(encoder(batch).float().cpu())

    z_all = torch.cat(z_parts, dim=0)


    if gpu_resident and x_gpu is not None:
        del x_gpu
    elif not gpu_resident and x_cpu is not None:
        del x_cpu
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


    z_np = z_all.numpy()
    h3_arr = np.array(data["h3_labels"])

    h3_to_idx = {}
    for i, h3_id in enumerate(h3_arr):
        if h3_id not in h3_to_idx:
            h3_to_idx[h3_id] = []
        h3_to_idx[h3_id].append(i)

    final_h3_list = sorted(h3_to_idx.keys())
    final_z = np.zeros((len(final_h3_list), z_np.shape[1]), dtype=np.float32)
    for i, h3_id in enumerate(final_h3_list):
        idxs = h3_to_idx[h3_id]
        final_z[i] = z_np[idxs].mean(axis=0)

    return final_z, final_h3_list


def save_results(city: str, final_z: np.ndarray, final_h3_list: list, modules: dict):
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, "streetview_embeddings.npy"), final_z)
    np.save(os.path.join(SAVE_DIR, "streetview_h3_order.npy"), np.array(final_h3_list))
    torch.save({
        "encoder": modules["encoder"].state_dict(),
        "predictor": modules["predictor"].state_dict(),
        "contrastive": modules["contrastive"].state_dict(),
        "reconstruction": modules["reconstruction"].state_dict(),
    }, os.path.join(SAVE_DIR, "streetview_model.pt"))


def main():
    global DATA_PATH, SAVE_DIR, CACHE_DIR, EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city", default=CITIES[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    EPOCHS = args.epochs
    DATA_PATH = str(args.data_dir.resolve())
    SAVE_DIR = str(args.output_dir.resolve())
    CACHE_DIR = str(args.output_dir.resolve() / "streetview_cache")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 70)
    print(f"   Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print("=" * 70)

    for city in (args.city,):
        data = load_streetview_data(city)
        modules = build_streetview_modules()
        final_z, final_h3_list = train_and_aggregate(city, data, modules)
        save_results(city, final_z, final_h3_list, modules)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

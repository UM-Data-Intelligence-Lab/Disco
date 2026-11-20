from copy import deepcopy

import torch


def train_model_downstream_moe(model, train_loader, val_loader,
                                criterion, device, config):
    epochs   = config.get("epochs",  150)
    patience = config.get("patience", 30)

    model     = model.to(device)
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config["optimizer"]]
    optimizer = optimizer_type(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config["eta_min"]
    )

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0

        for batch in train_loader:
            shared  = batch["shared"].to(device)
            functional = batch["functional"].to(device)
            visual  = batch["visual"].to(device)
            y   = batch["label"].to(device)
            if y.ndim == 2 and y.shape[-1] == 1:
                y = y.squeeze(-1)

            preds, _ = model(shared, functional, visual)
            if preds.shape[-1] == 1:
                preds = preds.squeeze(-1)

            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_gradient_norm"])
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        avg_train = train_loss / max(n_batches, 1)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                shared  = batch["shared"].to(device)
                functional = batch["functional"].to(device)
                visual  = batch["visual"].to(device)
                y   = batch["label"].to(device)
                if y.ndim == 2 and y.shape[-1] == 1:
                    y = y.squeeze(-1)
                preds, _ = model(shared, functional, visual)
                if preds.shape[-1] == 1:
                    preds = preds.squeeze(-1)
                val_loss += criterion(preds, y).item()
                n_val    += 1

        avg_val = val_loss / max(n_val, 1)
        scheduler.step()

        if epoch % 10 == 0 or epoch <= 3:
            print(f"      Epoch {epoch:3d}: train={avg_train:.4f}  "
                  f"val={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val_loss - config["early_stop_delta"]:
            best_val_loss = avg_val
            best_state    = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"   Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

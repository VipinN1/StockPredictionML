# train.py
import torch

def fit(
    model,
    train_loader,
    val_loader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    ckpt_path: str = "best_transformer.pth",
):
    """
    Train & validate `model` for `epochs` epochs, saving the best on-disk.

    Args:
      model        – your nn.Module
      train_loader – DataLoader for training
      val_loader   – DataLoader for validation
      epochs       – number of epochs
      optimizer    – torch optimizer
      criterion    – loss function
      device       – 'cpu' or 'cuda'
      ckpt_path    – filename to save best weights
    """
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # — TRAIN —
        model.train()
        total_train = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device).float(), yb.to(device).float()
            optimizer.zero_grad()
            preds = model(Xb)                 # → (batch, 1)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * Xb.size(0)
        avg_train = total_train / len(train_loader.dataset)

        # — VALIDATE —
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device).float(), yv.to(device).float()
                total_val += criterion(model(Xv), yv).item() * Xv.size(0)
        avg_val = total_val / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # — checkpoint the best model —
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), ckpt_path)
            print("  ↳ saved new best model")

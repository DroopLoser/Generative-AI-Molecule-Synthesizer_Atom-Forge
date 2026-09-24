
"""
Train the multi-property GNN on ZINC250k.

Improvements over original:
- StandardScaler normalizes all 6 targets
  so MolWt doesn't dominate the loss
- Scaler saved alongside model for correct inference
- Train / val split with per-property MAE logged
- Uses 6 atom features (in_channels=6)
- SELFIES decoded to SMILES for RDKit graph building
"""

import joblib
import numpy as np
import selfies as sf
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from data import load_zinc_selfies
from models.gnn import MolecularGNN
from utils.chemistry import PROPERTY_NAMES, smiles_to_graph


# -------------------------------------------------------
# Build graph dataset from SELFIES list
# -------------------------------------------------------
def build_dataset(selfies_list):
    dataset = []
    skipped = 0

    for s in selfies_list:
        try:
            smiles = sf.decoder(s)
        except Exception:
            skipped += 1
            continue

        g = smiles_to_graph(smiles)

        if g is not None:
            dataset.append(g)
        else:
            skipped += 1

    print(f"Valid graphs: {len(dataset)}  |  Skipped: {skipped}")
    return dataset


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Properties:", PROPERTY_NAMES)

    # Load data
    selfies_data = load_zinc_selfies(max_molecules=50000)

    print("Building molecular graphs...")
    dataset = build_dataset(selfies_data)

    if not dataset:
        raise ValueError("No valid molecules found.")

    # Train / val split — 90 / 10
    split     = int(0.9 * len(dataset))
    train_ds  = dataset[:split]
    val_ds    = dataset[split:]
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Fit StandardScaler on training targets only
    y_train = np.array(
        [g.y.numpy()[0] for g in train_ds]
    )
    scaler = StandardScaler()
    scaler.fit(y_train)

    # Scale targets for all graphs
    for g in dataset:
        g.y = torch.tensor(
            scaler.transform(g.y.numpy()),
            dtype=torch.float,
        )

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=256
    )

    # Model
    model = MolecularGNN(
        in_channels=6,
        hidden_dim=256,
        out_dim=len(PROPERTY_NAMES),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
        verbose=True,
    )

    num_epochs = 200

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    for epoch in range(num_epochs):

        # --- Train ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(
                batch.x,
                batch.edge_index,
                batch.batch,
            )

            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss  = 0.0
        all_pred  = []
        all_true  = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred  = model(
                    batch.x,
                    batch.edge_index,
                    batch.batch,
                )
                val_loss += F.mse_loss(pred, batch.y).item()
                all_pred.append(pred.cpu().numpy())
                all_true.append(batch.y.cpu().numpy())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        scheduler.step(avg_val)

        # Log per-property MAE every 20 epochs
        if epoch % 20 == 0:
            preds = np.concatenate(all_pred)
            trues = np.concatenate(all_true)
            mae   = np.abs(preds - trues).mean(axis=0)
            mae_str = " | ".join(
                f"{n}: {m:.3f}"
                for n, m in zip(PROPERTY_NAMES, mae)
            )
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"Train MSE: {avg_train:.5f} | "
                f"Val MSE: {avg_val:.5f}\n"
                f"  Val MAE (scaled) -> {mae_str}"
            )

    # -------------------------------------------------------
    # Save checkpoint + scaler
    # -------------------------------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "property_names":   PROPERTY_NAMES,
        "in_channels":      6,
        "hidden_dim":       256,
        "out_dim":          len(PROPERTY_NAMES),
    }, "gnn.pt")

    joblib.dump(scaler, "gnn_scaler.pkl")

    print("Saved GNN to gnn.pt")
    print("Saved scaler to gnn_scaler.pkl")


if __name__ == "__main__":
    main()
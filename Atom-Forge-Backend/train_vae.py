
"""
Train the SELFIES VAE on ZINC250k.

Improvements over original:
- KL annealing: beta warms up over first 10 epochs
  prevents posterior collapse
- Train / validation split (90/10)
- Validity rate logged every 5 epochs
- Gradient clipping for stable training
- ReduceLROnPlateau scheduler
"""

import torch
import selfies as sf
from rdkit import Chem
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import load_zinc_selfies
from utils.tokenizer import SelfiesTokenizer
from models.vae import SelfiesVAE, vae_loss


# -------------------------------------------------------
# Validity metric
# -------------------------------------------------------
def compute_validity(model, tokenizer, device, n=200):
    """
    Fraction of randomly generated molecules
    that decode to valid SMILES.
    """
    model.eval()
    valid = 0

    with torch.no_grad():
        for _ in range(n):
            z = torch.randn(
                1,
                model.latent_dim,
                device=device,
            )
            logits = model.decode(z)
            tokens = logits.argmax(-1)[0]
            selfies_str = tokenizer.decode(tokens.tolist())

            try:
                smiles = sf.decoder(selfies_str)
                if smiles and Chem.MolFromSmiles(smiles):
                    valid += 1
            except Exception:
                pass

    model.train()
    return valid / n


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Load data
    selfies_data = load_zinc_selfies(max_molecules=50000)

    # Tokenizer
    tokenizer = SelfiesTokenizer(selfies_data)
    print(f"Vocab size : {tokenizer.vocab_size}")
    print(f"Max length : {tokenizer.max_len}")

    # Encode all molecules
    print("Encoding molecules...")
    encoded = torch.stack(
        [tokenizer.encode(s) for s in selfies_data]
    )

    # Train / val split — 90 / 10
    full_ds   = TensorDataset(encoded)
    val_size  = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=256
    )

    print(f"Train: {train_size}  |  Val: {val_size}")

    # Model
    model = SelfiesVAE(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        embed_dim=128,
        hidden_dim=512,
        latent_dim=128,
        sos_idx=tokenizer.sos_idx,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5,
        verbose=True,
    )

    num_epochs   = 30
    beta_max     = 0.01
    anneal_epochs = 10

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------
    for epoch in range(num_epochs):

        # KL annealing — beta increases linearly
        # for the first anneal_epochs then stays at beta_max
        beta = min(
            beta_max,
            beta_max * (epoch + 1) / anneal_epochs,
        )

        # --- Train ---
        model.train()
        train_loss = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            logits, mu, logvar = model(batch_x)
            loss, recon, kl = vae_loss(
                logits, batch_x, mu, logvar, beta=beta
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                logits, mu, logvar = model(batch_x)
                loss, _, _ = vae_loss(
                    logits, batch_x, mu, logvar, beta=beta
                )
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        scheduler.step(avg_val)

        # Log validity every 5 epochs
        if (epoch + 1) % 5 == 0:
            validity = compute_validity(
                model, tokenizer, device, n=200
            )
            print(
                f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"Train: {avg_train:.4f} | "
                f"Val: {avg_val:.4f} | "
                f"Beta: {beta:.4f} | "
                f"Validity: {validity*100:.1f}%"
            )
        else:
            print(
                f"Epoch {epoch+1:02d}/{num_epochs} | "
                f"Train: {avg_train:.4f} | "
                f"Val: {avg_val:.4f} | "
                f"Beta: {beta:.4f}"
            )

    # -------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size":       tokenizer.vocab_size,
        "max_len":          tokenizer.max_len,
        "embed_dim":        128,
        "hidden_dim":       512,
        "latent_dim":       128,
        "sos_idx":          tokenizer.sos_idx,
        "idx_to_token":     tokenizer.idx_to_token,
    }, "vae.pt")

    print("Saved VAE to vae.pt")


if __name__ == "__main__":
    main()
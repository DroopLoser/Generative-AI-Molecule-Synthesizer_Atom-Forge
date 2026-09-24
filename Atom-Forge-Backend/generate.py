# generate.py
"""
Model loading utilities shared by both
the CLI (cli.py) and the API (fastapi_app.py).
"""

import torch
from models.vae import SelfiesVAE
from models.gnn import MolecularGNN
from utils.chemistry import PROPERTY_NAMES


def load_vae(device):
    map_loc = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("vae.pt", map_location=map_loc)

    model = SelfiesVAE(
        vocab_size=ckpt["vocab_size"],
        max_len=ckpt["max_len"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
        sos_idx=ckpt.get("sos_idx", 1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["idx_to_token"]


def load_gnn(device):
    map_loc = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("gnn.pt", map_location=map_loc)

    model = MolecularGNN(
        in_channels=ckpt.get("in_channels", 6),
        hidden_dim=ckpt.get("hidden_dim", 256),
        out_dim=ckpt.get("out_dim", len(PROPERTY_NAMES)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model
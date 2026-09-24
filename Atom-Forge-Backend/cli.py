# cli.py
"""
Command-line molecule generation.

Usage:
    python cli.py --property QED --target 0.8
    python cli.py --property MolWt --target 300
"""

import argparse
import joblib
import torch

from generate import load_vae, load_gnn
from utils.chemistry import PROPERTY_NAMES
from utils.optimization import optimize_latent_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", dest="selected_property",
                        default="QED")
    parser.add_argument("--target",   type=float, required=True)
    parser.add_argument("--random",   type=int, default=500)
    parser.add_argument("--local",    type=int, default=500)
    args = parser.parse_args()

    if args.selected_property not in PROPERTY_NAMES:
        raise ValueError(
            f"Unknown property '{args.selected_property}'.\n"
            f"Choose from: {PROPERTY_NAMES}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    vae_model, idx_to_token = load_vae(device)
    gnn_model = load_gnn(device)
    scaler = joblib.load("gnn_scaler.pkl")

    results = optimize_latent_space(
        vae_model=vae_model,
        gnn_model=gnn_model,
        scaler=scaler,
        idx_to_token=idx_to_token,
        target_property=args.target,
        selected_property=args.selected_property,
        num_random=args.random,
        num_local=args.local,
        device=device,
    )

    print("\nTop Candidate Molecules")
    print("-" * 72)

    if not results:
        print("No valid molecules generated.")
        return

    for i, r in enumerate(results, 1):
        print(
            f"{i:2d}. {r['smiles']:35s} | "
            f"{r['selected_property']}: "
            f"{r['predicted_property']:.4f} | "
            f"Error: {r['error']:.4f}"
        )


if __name__ == "__main__":
    main()
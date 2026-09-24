# fix_scaler.py
"""
Re-fits and re-saves the StandardScaler using the
current local sklearn version.
Run this once after moving from Colab to local.
"""

import joblib
import numpy as np
import selfies as sf
import torch
from sklearn.preprocessing import StandardScaler

from data import load_zinc_selfies
from utils.chemistry import PROPERTY_NAMES, smiles_to_graph


def main():
    print("Loading data...")
    selfies_data = load_zinc_selfies(max_molecules=50000)

    print("Building targets...")
    y_list = []

    for s in selfies_data:
        try:
            smiles = sf.decoder(s)
        except Exception:
            continue

        g = smiles_to_graph(smiles)
        if g is not None:
            y_list.append(g.y.numpy()[0])

    y_array = np.array(y_list)
    print(f"Collected {len(y_array)} molecules.")

    # Re-fit scaler with local sklearn version
    scaler = StandardScaler()
    scaler.fit(y_array)

    joblib.dump(scaler, "gnn_scaler.pkl")
    print("Saved new gnn_scaler.pkl with local sklearn version.")
    print(f"sklearn version used: {joblib.__version__}")


if __name__ == "__main__":
    main()
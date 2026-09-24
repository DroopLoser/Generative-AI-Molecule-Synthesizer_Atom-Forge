
"""
ZINC250k dataset loader with SELFIES conversion.

Improvements over original:
- Uses ZINC250k (drug-like molecules) instead of QM9
- Converts SMILES to SELFIES for guaranteed valid generation
- Does NOT run at import time — call load_zinc_selfies() explicitly
"""

import os
import requests
import pandas as pd
import selfies as sf
from rdkit import Chem

ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/"
    "chemical_vae/master/models/zinc_properties/"
    "250k_rndm_zinc_drugs_clean_3.csv"
)


def load_zinc_selfies(max_molecules=50000, cache_path="data/zinc250k.csv"):
    """
    Download ZINC250k once, convert SMILES to SELFIES,
    return a deduplicated list of valid SELFIES strings.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.exists(cache_path):
        print("Downloading ZINC250k dataset...")
        response = requests.get(ZINC250K_URL, timeout=120)
        response.raise_for_status()
        with open(cache_path, "w") as f:
            f.write(response.text)
        print("Download complete.")

    df = pd.read_csv(cache_path)

    # Detect SMILES column flexibly
    smiles_col = None
    for col in df.columns:
        if col.lower() in ("smiles", "smile"):
            smiles_col = col
            break
    if smiles_col is None:
        smiles_col = df.columns[0]

    selfies_list = []
    seen = set()

    for smiles in df[smiles_col].tolist():
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            selfies_str = sf.encoder(canonical)
            if not selfies_str or selfies_str in seen:
                continue
            seen.add(selfies_str)
            selfies_list.append(selfies_str)
        except Exception:
            continue

        if len(selfies_list) >= max_molecules:
            break

    print(f"Loaded {len(selfies_list)} SELFIES strings from ZINC250k.")
    return selfies_list
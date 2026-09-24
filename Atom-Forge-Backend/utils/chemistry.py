
"""
Chemistry helpers: validation, descriptors, graph conversion, rendering.

Improvements over original:
- PROPERTY_NAMES and smiles_to_graph centralized here
  so all files share one source of truth
- atom_features uses 6 chemically meaningful features
  instead of just atomic number
- molecule_to_image_bytes returns PNG bytes
  instead of saving temp files to disk
"""

from io import BytesIO

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Draw import MolToImage
from torch_geometric.data import Data
from rdkit.Chem import AllChem


# -------------------------------------------------------
# Shared constants
# -------------------------------------------------------
PROPERTY_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "QED"]


# -------------------------------------------------------
# Validation
# -------------------------------------------------------
def is_valid_smiles(smiles):
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_3d_sdf(smiles):
    """
    Generate 3D coordinates for a molecule
    and return as an SDF string for 3Dmol.js rendering.
    Returns None if 3D generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # Add explicit hydrogens for better 3D shape
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates using ETKDGv3
        # (best available conformer algorithm in RDKit)
        result = AllChem.EmbedMolecule(
            mol,
            AllChem.ETKDGv3(),
        )

        if result != 0:
            # ETKDGv3 failed, try random coords fallback
            result = AllChem.EmbedMolecule(
                mol,
                AllChem.ETKDG(),
            )

        if result != 0:
            return None

        # Optimize geometry with MMFF force field
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            # Optimization failed but coords still usable
            pass

        return Chem.MolToMolBlock(mol)

    except Exception:
        return None

# -------------------------------------------------------
# Descriptors
# -------------------------------------------------------
def property_by_name(smiles, property_name="MolWt"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    p = property_name.lower()
    if p == "molwt":  return Descriptors.MolWt(mol)
    if p == "logp":   return Descriptors.MolLogP(mol)
    if p == "tpsa":   return Descriptors.TPSA(mol)
    if p == "hbd":    return Descriptors.NumHDonors(mol)
    if p == "hba":    return Descriptors.NumHAcceptors(mol)
    if p == "qed":    return QED.qed(mol)
    raise ValueError(f"Unsupported property: {property_name}")


# -------------------------------------------------------
# Atom features — 6 chemically meaningful features
# -------------------------------------------------------
def atom_features(atom):
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumImplicitHs()),
        float(int(atom.GetIsAromatic())),
        float(int(atom.IsInRing())),
    ]


# -------------------------------------------------------
# Graph conversion
# -------------------------------------------------------
def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a PyG Data object.
    Returns None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float,
    )

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    values = []
    for name in PROPERTY_NAMES:
        v = property_by_name(smiles, name)
        if v is None:
            return None
        values.append(float(v))

    y = torch.tensor([values], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


# -------------------------------------------------------
# Rendering — returns PNG bytes, no temp files
# -------------------------------------------------------
def molecule_to_image_bytes(smiles, size=(300, 300)):
    """
    Render a molecule to PNG bytes.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
# utils/naming.py
"""
Molecule naming pipeline with 4 layers:

1. CIR     — IUPAC names, works for novel structures
2. PubChem — common names + IUPAC fallback
3. InChI   — last resort systematic identifier (RDKit, local)
4. Formula — always filled (RDKit, local)

CIR and PubChem run simultaneously per molecule.
"""

import asyncio
import urllib.parse
import httpx
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, inchi


# -------------------------------------------------------
# Local helpers — RDKit, no API needed
# -------------------------------------------------------
def get_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


def get_inchi(smiles):
    """
    Last resort systematic identifier.
    Not a true IUPAC name but unique,
    always works locally for any valid structure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        result = inchi.MolToInchi(mol)
        return result if result else ""
    except Exception:
        return ""


# -------------------------------------------------------
# Layer 1 — NCI CIR (async)
# Best for novel generated molecules
# -------------------------------------------------------
async def get_iupac_from_cir_async(smiles, client):
    try:
        encoded  = urllib.parse.quote(smiles, safe="")
        url      = (
            f"https://cactus.nci.nih.gov/chemical"
            f"/structure/{encoded}/iupac_name"
        )
        response = await client.get(url, timeout=10.0)

        if response.status_code != 200:
            return ""

        name = response.text.strip()

        if not name or "<" in name or "Page not found" in name:
            return ""

        return name

    except Exception:
        return ""


# -------------------------------------------------------
# Layer 2 — PubChem (async)
# Best for common names of known molecules
# -------------------------------------------------------
async def get_names_from_pubchem_async(smiles, client):
    """Returns (iupac_name, common_name)"""
    try:
        response = await client.post(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
            "/compound/smiles/property"
            "/IUPACName,MolecularFormula,Title/JSON",
            data={"smiles": smiles},
            timeout=10.0,
        )

        if response.status_code != 200:
            return "", ""

        props = (
            response.json()
            .get("PropertyTable", {})
            .get("Properties", [{}])[0]
        )

        return (
            props.get("IUPACName", ""),
            props.get("Title", ""),
        )

    except Exception:
        return "", ""


# -------------------------------------------------------
# Main async entry point
# -------------------------------------------------------
async def get_molecule_names_async(smiles, client):
    """
    Priority order:
    1. CIR IUPAC       (novel + known molecules)
    2. PubChem IUPAC   (known molecules only)
    3. InChI           (last resort, always local)
    4. Formula         (always filled, local)

    CIR and PubChem run simultaneously.
    """
    formula = get_formula(smiles)
    inchi_str  = get_inchi(smiles)

    # CIR and PubChem fire at the same time
    cir_task     = get_iupac_from_cir_async(smiles, client)
    pubchem_task = get_names_from_pubchem_async(smiles, client)

    iupac_from_cir, (pubchem_iupac, common_name) = (
        await asyncio.gather(cir_task, pubchem_task)
    )

    # Build iupac_name — first source that returns something
    iupac_name = (
        iupac_from_cir          # Layer 1 — CIR
        or pubchem_iupac        # Layer 2 — PubChem
        or ""    # Layer 3 — InChI fallback
    )

    # Display name — prefer human-readable common name
    name = common_name or iupac_name or formula

    return {
        "name":            name,
        "systematic_name": iupac_name,
        "iupac_name":      iupac_name,
        "common_name":     common_name,
        "formula":         formula,
        "inchi":           inchi_str,
    }


# -------------------------------------------------------
# Sync wrapper — kept for backward compatibility
# (used by fix_scaler.py and any sync code)
# -------------------------------------------------------
def get_molecule_names(smiles):
    async def _run():
        async with httpx.AsyncClient() as client:
            return await get_molecule_names_async(smiles, client)
    return asyncio.run(_run())
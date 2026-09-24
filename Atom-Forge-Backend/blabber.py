# debug_naming.py
from utils.naming import (
    get_iupac_cirpy,
    get_iupac_cir_direct,
    get_names_pubchempy,
)

smiles = "CC[NH+]1CCS(=O)(=O)c2ccccc21"

print("Layer 1 - cirpy     :", repr(get_iupac_cirpy(smiles)))
print("Layer 2 - CIR direct:", repr(get_iupac_cir_direct(smiles)))
print("Layer 3 - pubchempy :", repr(get_names_pubchempy(smiles)))

"""
Two-phase latent-space optimization.

Improvements over original:
- Phase 1: random search across latent space
- Phase 2: local refinement around top-K candidates
  by adding small Gaussian noise to best z vectors
  this is much better than pure random search
- Scaler-aware: inverse transforms GNN predictions
  back to real units before comparing to target
- No circular imports: everything from utils.chemistry
"""

import torch
import selfies as sf

from utils.chemistry import (
    PROPERTY_NAMES,
    smiles_to_graph,
    is_valid_smiles,
    canonicalize_smiles,
)




# -------------------------------------------------------
# Decode token IDs → SELFIES → SMILES
# -------------------------------------------------------
def tokens_to_smiles(tokens, idx_to_token):
    parts = []
    for idx in tokens.tolist():
        if idx <= 1:
            continue
        tok = (
            idx_to_token[idx]
            if idx < len(idx_to_token)
            else None
        )
        if tok and tok not in ("<PAD>", "<SOS>"):
            parts.append(tok)

    selfies_str = "".join(parts)

    try:
        return sf.decoder(selfies_str)
    except Exception:
        return None


# -------------------------------------------------------
# Predict a single property with GNN + scaler
# -------------------------------------------------------
def predict_property(
    smiles,
    gnn_model,
    scaler,
    selected_property,
    device,
):
    prop_idx = PROPERTY_NAMES.index(selected_property)

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    graph = graph.to(device)
    batch = torch.zeros(
        graph.x.size(0),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        pred_scaled = gnn_model(
            graph.x,
            graph.edge_index,
            batch,
        )

    # Inverse transform back to real units
    pred_real = scaler.inverse_transform(
        pred_scaled.cpu().numpy()
    )

    return float(pred_real[0, prop_idx])


# -------------------------------------------------------
# Decode one z vector → scored result dict or None
# -------------------------------------------------------
def _decode_and_score(
    z,
    vae_model,
    gnn_model,
    scaler,
    idx_to_token,
    selected_property,
    target_value,
    seen,
    device,
):
    logits = vae_model.decode(z)
    tokens = logits.argmax(-1)[0]
    smiles = tokens_to_smiles(tokens, idx_to_token)

    if not smiles:
        return None, None

    if not is_valid_smiles(smiles):
        return None, None

    smiles = canonicalize_smiles(smiles)
    if smiles is None or smiles in seen:
        return None, None

    seen.add(smiles)

    predicted = predict_property(
        smiles,
        gnn_model,
        scaler,
        selected_property,
        device,
    )

    if predicted is None:
        return None, None

    result = {
        "smiles":               smiles,
        "selected_property":    selected_property,
        "predicted_property":   predicted,
        "target_property":      target_value,
        "error":                abs(predicted - target_value),
    }

    return result, z.clone()


# -------------------------------------------------------
# Main entry point
# -------------------------------------------------------
def optimize_latent_space(
    vae_model,
    gnn_model,
    scaler,
    idx_to_token,
    target_property,
    selected_property="QED",
    num_random=500,
    num_local=500,
    top_k_seeds=5,
    noise_scale=0.3,
    device="cpu",
):
    """
    Two-phase search in VAE latent space.

    Phase 1 — Random search:
        Sample `num_random` random latent vectors,
        decode each, validate, score with GNN.

    Phase 2 — Local refinement:
        Take the top-K results from phase 1,
        perturb their z vectors with Gaussian noise,
        decode and score again.

    Returns top-10 unique candidates sorted by
    closeness to target value.
    """

    if selected_property not in PROPERTY_NAMES:
        raise ValueError(
            f"Unknown property '{selected_property}'. "
            f"Choose from {PROPERTY_NAMES}."
        )

    vae_model.eval()
    gnn_model.eval()

    results  = []
    seed_zs  = []       # (error, z) pairs for phase 2
    seen     = set()

    # -------------------------------------------------------
    # Phase 1 — Random search
    # -------------------------------------------------------
    print(f"Phase 1: random search ({num_random} samples)...")

    with torch.no_grad():
        for i in range(num_random):
            z = torch.randn(
                1,
                vae_model.latent_dim,
                device=device,
            )

            result, z_clone = _decode_and_score(
                z, vae_model, gnn_model, scaler,
                idx_to_token, selected_property,
                target_property, seen, device,
            )

            if result:
                results.append(result)
                seed_zs.append((result["error"], z_clone))

    print(f"Phase 1 hits: {len(results)}")

    # -------------------------------------------------------
    # Phase 2 — Local refinement around top-K seeds
    # -------------------------------------------------------
    seed_zs.sort(key=lambda x: x[0])
    top_seeds = [z for _, z in seed_zs[:top_k_seeds]]

    if top_seeds:
        n_per_seed = max(1, num_local // len(top_seeds))
        print(
            f"Phase 2: local refinement "
            f"({len(top_seeds)} seeds × {n_per_seed} samples)..."
        )

        with torch.no_grad():
            for seed_z in top_seeds:
                for _ in range(n_per_seed):
                    z_perturbed = (
                        seed_z
                        + noise_scale
                        * torch.randn_like(seed_z)
                    )

                    result, _ = _decode_and_score(
                        z_perturbed, vae_model, gnn_model, scaler,
                        idx_to_token, selected_property,
                        target_property, seen, device,
                    )

                    if result:
                        results.append(result)

    print(f"Total candidates found: {len(results)}")

    # Sort by closeness to target and return top 10
    results.sort(key=lambda x: x["error"])
    return results[:10]
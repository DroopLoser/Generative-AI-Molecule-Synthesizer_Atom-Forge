
"""
Autoregressive SELFIES Variational Autoencoder.

Improvements over original:
- Decoder is now autoregressive (step-by-step GRU)
  instead of repeating the same hidden state every step
- Teacher forcing during training for stable learning
- Greedy decoding during generation
- 2-layer GRU encoder and decoder for more capacity
- Gradient clipping friendly (no exploding hidden states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfiesVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim=128,
        hidden_dim=512,
        latent_dim=128,
        sos_idx=1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.latent_dim = latent_dim
        self.sos_idx    = sos_idx

        # Shared embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        # Encoder — 2-layer GRU
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Latent distribution
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder initial hidden state
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

        # Decoder — 2-layer GRU
        self.decoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------
    def encode(self, x):
        emb = self.embedding(x)
        _, h = self.encoder(emb)
        h = h[-1]                       # take last layer hidden state
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # --------------------------------------------------
    # Reparameterization
    # --------------------------------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # --------------------------------------------------
    # Decoder
    # --------------------------------------------------
    def decode(self, z, teacher_tokens=None):
        """
        z               : (B, latent_dim)
        teacher_tokens  : (B, seq_len) passed during training
                          None during generation
        returns logits  : (B, seq_len, vocab_size)
        """
        B = z.size(0)

        # Project z into 2-layer hidden state
        h0 = self.fc_decode(z)                  # (B, H)
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)   # (2, B, H)

        seq_len = (
            teacher_tokens.size(1)
            if teacher_tokens is not None
            else self.max_len
        )

        # Start every sequence with <SOS>
        tok = torch.full(
            (B, 1),
            self.sos_idx,
            dtype=torch.long,
            device=z.device,
        )

        outputs = []
        h = h0

        for t in range(seq_len):
            emb   = self.embedding(tok)         # (B, 1, E)
            out, h = self.decoder(emb, h)       # (B, 1, H)
            logit  = self.output_layer(out)     # (B, 1, V)
            outputs.append(logit)

            if teacher_tokens is not None:
                # Teacher forcing: feed actual token as next input
                tok = teacher_tokens[:, t:t+1]
            else:
                # Greedy: feed predicted token as next input
                tok = logit.argmax(-1)

        return torch.cat(outputs, dim=1)        # (B, seq_len, V)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        logits     = self.decode(z, teacher_tokens=x)
        return logits, mu, logvar


# -------------------------------------------------------
# Loss Function
# -------------------------------------------------------
def vae_loss(logits, targets, mu, logvar, beta=0.01):
    """
    Reconstruction loss (cross entropy) + KL divergence.
    beta controls the weight of KL — kept small and annealed.
    """
    V = logits.size(-1)

    recon = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=0,
    )

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon + beta * kl, recon, kl
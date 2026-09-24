
"""
SELFIES token-level tokenizer.

Improvements over original:
- Splits on SELFIES tokens like [C], [=O], [Branch1]
  instead of individual characters
- Preserves the validity guarantee of SELFIES
- Adds <SOS> token for autoregressive decoding
"""

import torch
import selfies as sf


class SelfiesTokenizer:
    def __init__(self, selfies_list):
        # Build vocabulary from SELFIES tokens
        tokens = set()
        for s in selfies_list:
            for tok in sf.split_selfies(s):
                tokens.add(tok)

        # Reserved tokens
        self.token_to_idx = {
            "<PAD>": 0,
            "<SOS>": 1,
        }

        for tok in sorted(tokens):
            self.token_to_idx[tok] = len(self.token_to_idx)

        # Reverse mapping
        self.idx_to_token = [None] * len(self.token_to_idx)
        for tok, idx in self.token_to_idx.items():
            self.idx_to_token[idx] = tok

        self.vocab_size = len(self.token_to_idx)
        self.pad_idx    = 0
        self.sos_idx    = 1
        self.max_len    = max(
            len(list(sf.split_selfies(s))) for s in selfies_list
        )

    def encode(self, selfies_str):
        """
        Convert a SELFIES string to a fixed-length tensor.
        Unknown tokens are skipped.
        """
        tokens = [
            self.token_to_idx[tok]
            for tok in sf.split_selfies(selfies_str)
            if tok in self.token_to_idx
        ]
        # Pad to max length
        while len(tokens) < self.max_len:
            tokens.append(self.pad_idx)

        return torch.tensor(
            tokens[:self.max_len],
            dtype=torch.long,
        )

    def decode(self, token_ids):
        """
        Convert token IDs back to a SELFIES string.
        """
        parts = []
        for idx in token_ids:
            idx = int(idx)
            if idx in (self.pad_idx, self.sos_idx):
                continue
            tok = self.idx_to_token[idx] if idx < len(self.idx_to_token) else None
            if tok is None or tok in ("<PAD>", "<SOS>"):
                continue
            parts.append(tok)
        return "".join(parts)
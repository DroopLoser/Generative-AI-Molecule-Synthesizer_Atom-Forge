
"""
Graph Attention Network for multi-property molecular prediction.

Improvements over original:
- GATConv instead of GCNConv
  (attention learns which neighboring atoms matter more)
- 3 conv layers instead of 2 for deeper representation
- in_channels=6 to use all 6 atom features
- Extra fully connected layer before output
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class MolecularGNN(nn.Module):
    def __init__(
        self,
        in_channels=6,
        hidden_dim=256,
        out_dim=6,
        heads=4,
    ):
        super().__init__()

        # 3 graph attention layers
        # concat=False averages over heads → output stays hidden_dim
        self.conv1 = GATConv(
            in_channels,
            hidden_dim,
            heads=heads,
            concat=False,
        )
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
        )
        self.conv3 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
        )

        # MLP head
        self.fc1    = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch):
        # Graph attention layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Pool all node embeddings into one graph embedding
        x = global_mean_pool(x, batch)

        # MLP
        x = F.relu(self.fc1(self.dropout(x)))

        return self.fc_out(x)
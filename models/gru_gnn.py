"""
GRU-GNN Fusion Module
Implements temporal awareness and graph structural fusion
Based on Algorithm 2 and Section 3.2 of the paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class GRULayer(nn.Module):
    """
    GRU layer for capturing temporal features
    Corresponds to Equations (1)-(4) in the paper
    """

    def __init__(self, input_dim, hidden_dim):
        super(GRULayer, self).__init__()
        self.hidden_dim = hidden_dim

        # GRU parameters
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)  # reset gate
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)  # update gate
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)  # candidate hidden state

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: Input at current time step (batch, input_dim)
            h_prev: Previous hidden state (batch, hidden_dim)

        Returns:
            h_t: Current hidden state (batch, hidden_dim)
        """
        combined = torch.cat([x_t, h_prev], dim=1)

        # Eq. (1): Reset gate
        r_t = torch.sigmoid(self.W_r(combined))

        # Eq. (2): Update gate
        z_t = torch.sigmoid(self.W_z(combined))

        # Eq. (3): Candidate hidden state
        combined_reset = torch.cat([x_t, r_t * h_prev], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))

        # Eq. (4): Update hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer
    Corresponds to Equation (5) in the paper
    """

    def __init__(self, in_dim, out_dim):
        super(GraphConvLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, g, h):
        """
        Args:
            g: DGL graph
            h: Node features (N, in_dim)

        Returns:
            h_new: Updated node features (N, out_dim)
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h_new = self.W(h_neigh)
            return F.relu(h_new)


class GRU_GNN(nn.Module):
    """
    GRU-GNN Fusion Network
    Implements Algorithm 2 in the paper
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, fusion_alpha=0.5):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: GRU hidden dimension
            output_dim: Output feature dimension
            num_layers: Number of GNN layers
            fusion_alpha: Fusion weight (Eq. 6)
        """
        super(GRU_GNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fusion_alpha = fusion_alpha

        # GRU layer
        self.gru = GRULayer(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
            else:
                self.gnn_layers.append(GraphConvLayer(hidden_dim, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, node_features, timestamps=None):
        """
        Args:
            g: DGL heterogeneous graph
            node_features: Node features (N, F)
            timestamps: Optional timestamps

        Returns:
            h_fused: Fused node embeddings (N, output_dim)
        """
        num_nodes = node_features.shape[0]

        # Initialize GRU hidden state
        h_gru = torch.zeros(num_nodes, self.hidden_dim).to(node_features.device)

        # Temporal modeling via GRU
        h_gru = self.gru(node_features, h_gru)

        # Structural modeling via GNN
        h_gnn = h_gru
        for gnn_layer in self.gnn_layers:
            h_gnn = gnn_layer(g, h_gnn)

        # Eq. (6): Weighted fusion
        h_fused = self.fusion_alpha * h_gru + (1 - self.fusion_alpha) * h_gnn

        # Output projection
        h_out = self.output_proj(h_fused)

        return h_out

    def get_temporal_features(self, node_features, h_prev):
        """Get temporal features (used for incremental learning)"""
        return self.gru(node_features, h_prev)

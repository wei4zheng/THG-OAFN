"""
Multi-layer Attention Mechanism Module
Implements relation fusion, neighborhood fusion, and information perception.
Based on Algorithm 4 and Section 3.3 of the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationFusionLayer(nn.Module):
    """
    Relation Fusion Layer
    Corresponds to Eq. (20): Fusing multiple relation types.
    """

    def __init__(self, num_relations):
        super(RelationFusionLayer, self).__init__()
        # Learnable weight for each relation type
        self.relation_weights = nn.Parameter(torch.randn(num_relations))

    def forward(self, adj_matrices):
        """
        Args:
            adj_matrices: List of adjacency matrices for different relations [(N,N), ...] or None

        Returns:
            fused_adj: Fused adjacency matrix (N, N) - sparse format
        """
        if adj_matrices is None or len(adj_matrices) == 0:
            # If no adjacency matrices are provided, return None
            # The following layers should directly use the DGL graph
            return None

        # Eq. (20): A_fused = Σ_r σ(w_r) ⊙ A^(r)
        weights = torch.sigmoid(self.relation_weights)

        # Sparse or dense summation
        if torch.is_tensor(adj_matrices[0]) and adj_matrices[0].is_sparse:
            fused_adj = weights[0] * adj_matrices[0]
            for i in range(1, len(adj_matrices)):
                fused_adj = fused_adj + weights[i] * adj_matrices[i]
        else:
            # Dense matrix version – may cause OOM for large graphs
            fused_adj = torch.zeros_like(adj_matrices[0])
            for i, adj in enumerate(adj_matrices):
                fused_adj += weights[i] * adj

        return fused_adj


class NeighborhoodFusionLayer(nn.Module):
    """
    Neighborhood Fusion Layer
    Corresponds to Eq. (21)-(22): Multi-hop neighborhood aggregation.
    """

    def __init__(self, hidden_dim, num_hops=2, num_heads=8):
        super(NeighborhoodFusionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads

        # Multi-head attention parameters
        self.W_h = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)
        ])

        # Attention weight vectors
        self.a = nn.ParameterList([
            nn.Parameter(torch.randn(2 * hidden_dim, 1)) for _ in range(num_heads)
        ])

        # Cross-hop fusion weights
        self.hop_weights = nn.Parameter(torch.randn(num_hops))

    def forward(self, g, node_features):
        """
        Args:
            g: DGL graph
            node_features: Node features (N, hidden_dim)

        Returns:
            h_fused: Fused node features (N, hidden_dim)
        """
        num_nodes = node_features.shape[0]
        device = node_features.device

        # Store embeddings of each hop
        hop_embeddings = []
        current_features = node_features

        for k in range(self.num_hops):
            # Eq. (21): Compute attention scores
            # a_vu^(k) = softmax(a^T[Wh_v||Wh_u])
            hop_embedding = self._aggregate_neighbors(
                g, current_features, k
            )
            hop_embeddings.append(hop_embedding)

            # Update features for the next hop
            current_features = hop_embedding

        # Eq. (23): Cross-hop fusion
        # h_v^multi = Σ_k β^(k) h_v^(k)
        hop_weights = F.softmax(self.hop_weights, dim=0)
        h_fused = torch.zeros_like(node_features)

        for k, hop_emb in enumerate(hop_embeddings):
            h_fused += hop_weights[k] * hop_emb

        return h_fused

    def _aggregate_neighbors(self, g, node_features, hop_idx):
        """Aggregate neighbor information (single-hop)."""
        num_nodes = node_features.shape[0]
        device = node_features.device

        head_outputs = []

        for head_idx in range(self.num_heads):
            # Feature transformation
            h_transformed = self.W_h[head_idx](node_features)

            # Compute attention-weighted neighbor aggregation
            with g.local_scope():
                g.ndata['h'] = h_transformed

                # Simplified attention: mean pooling over neighbors
                g.update_all(
                    message_func=lambda edges: {'m': edges.src['h']},
                    reduce_func=lambda nodes: {'h_agg': torch.mean(nodes.mailbox['m'], dim=1)}
                )

                h_agg = g.ndata.get('h_agg', torch.zeros_like(h_transformed))

            head_outputs.append(h_agg)

        # Combine multi-head outputs
        h_out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)

        return h_out


class InformationPerceptionLayer(nn.Module):
    """
    Information Perception Layer
    Corresponds to Eq. (24)-(25): Multi-head attention and gating mechanism.
    """

    def __init__(self, hidden_dim, num_heads=8):
        super(InformationPerceptionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-head transformations
        self.W_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)
        ])

        # Gating vector — Eq. (25)
        self.gate_vector = nn.Parameter(torch.randn(num_heads))

    def forward(self, h_multi):
        """
        Args:
            h_multi: Multi-hop fused features (N, hidden_dim)

        Returns:
            h_final: Final node representation (N, hidden_dim)
        """
        # Eq. (24): Multi-head transformation
        # z_v^h = W_h h_v^multi
        head_outputs = []

        for head_idx in range(self.num_heads):
            z_h = self.W_heads[head_idx](h_multi)
            head_outputs.append(z_h)

        # Eq. (25): Gated fusion
        # h_v^final = Σ_h softmax(g)_h · z_v^h
        gate_weights = F.softmax(self.gate_vector, dim=0)
        h_final_list = []

        for h in range(self.num_heads):
            h_final_list.append(gate_weights[h] * head_outputs[h])

        h_final = torch.cat(h_final_list, dim=1)

        return h_final


class MultiLayerAttention(nn.Module):
    """
    Multi-Layer Attention Mechanism
    Integrates relation fusion, neighborhood fusion, and information perception.
    Corresponds to Algorithm 4 in the paper.
    """

    def __init__(self, hidden_dim, num_relations=4, num_hops=2, num_heads=8):
        super(MultiLayerAttention, self).__init__()

        # 1. Relation Fusion Layer
        self.relation_fusion = RelationFusionLayer(num_relations)

        # 2. Neighborhood Fusion Layer
        self.neighborhood_fusion = NeighborhoodFusionLayer(
            hidden_dim, num_hops, num_heads
        )

        # 3. Information Perception Layer
        self.information_perception = InformationPerceptionLayer(
            hidden_dim, num_heads
        )

    def forward(self, g, node_features, adj_matrices):
        """
        Args:
            g: DGL graph
            node_features: Node features (N, hidden_dim)
            adj_matrices: List of adjacency matrices for different relations (can be None)

        Returns:
            h_final: Final node representation (N, hidden_dim)
        """
        # Step 1: Relation Fusion
        fused_adj = self.relation_fusion(adj_matrices) if adj_matrices is not None else None

        # Step 2: Neighborhood Fusion (directly uses DGL graph, no adjacency matrix needed)
        h_multi = self.neighborhood_fusion(g, node_features)

        # Step 3: Information Perception
        h_final = self.information_perception(h_multi)

        return h_final

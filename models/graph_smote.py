"""
GraphSMOTE Oversampling Module
Implements topology-preserving oversampling
Based on Algorithm 3 and Section 3.2 of the paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class GraphSMOTE(nn.Module):
    """
    Graph-based SMOTE Oversampling
    Corresponds to Equations (12)-(19) in the paper
    """

    def __init__(self, embedding_dim, k_neighbors=5):
        """
        Args:
            embedding_dim: Node embedding dimension
            k_neighbors: Number of nearest neighbors for SMOTE
        """
        super(GraphSMOTE, self).__init__()
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors

        # Attribute completion network (HGNNAC)
        self.attribute_completion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # Edge reconstruction decoder (weighted inner product)
        self.edge_decoder_weight = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

    def forward(self, node_embeddings, labels, adj_matrix, oversample_ratio=2.0):
        """
        Args:
            node_embeddings: Node embeddings (N, embedding_dim)
            labels: Node labels (N,)
            adj_matrix: Adjacency matrix (N, N)
            oversample_ratio: Oversampling ratio

        Returns:
            new_embeddings: Node embeddings with synthetic nodes
            new_labels: Updated labels
            new_adj: Updated adjacency matrix
        """
        device = node_embeddings.device
        num_nodes = node_embeddings.shape[0]

        # Separate minority (fraud) and majority (normal) classes
        fraud_mask = labels == 1
        fraud_embeddings = node_embeddings[fraud_mask]
        num_fraud = fraud_embeddings.shape[0]

        if num_fraud == 0:
            return node_embeddings, labels, adj_matrix

        # Number of synthetic samples to generate
        num_to_generate = int(num_fraud * (oversample_ratio - 1))
        if num_to_generate <= 0:
            return node_embeddings, labels, adj_matrix

        # Eq. (14): Nearest neighbors search
        fraud_embeddings_np = fraud_embeddings.detach().cpu().numpy()
        knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, num_fraud))
        knn.fit(fraud_embeddings_np)

        new_embeddings_list = []
        for i in range(num_to_generate):
            # Randomly select a minority node
            idx = np.random.randint(0, num_fraud)
            base_embedding = fraud_embeddings[idx:idx + 1]

            # Find its nearest neighbor
            distances, indices = knn.kneighbors(
                base_embedding.detach().cpu().numpy(),
                n_neighbors=min(2, num_fraud)
            )
            neighbor_idx = indices[0, 1] if indices.shape[1] > 1 else indices[0, 0]
            neighbor_embedding = fraud_embeddings[neighbor_idx:neighbor_idx + 1]

            # Eq. (15): SMOTE interpolation
            delta = torch.rand(1).to(device)
            new_embedding = (1 - delta) * base_embedding + delta * neighbor_embedding

            # Eq. (13): Attribute completion
            new_embedding = self.attribute_completion(new_embedding)

            new_embeddings_list.append(new_embedding)

        # Combine generated samples
        if len(new_embeddings_list) > 0:
            new_fraud_embeddings = torch.cat(new_embeddings_list, dim=0)
            new_embeddings = torch.cat([node_embeddings, new_fraud_embeddings], dim=0)

            # Append new labels
            new_fraud_labels = torch.ones(num_to_generate, dtype=labels.dtype, device=device)
            new_labels = torch.cat([labels, new_fraud_labels], dim=0)

            # Eq. (16)-(19): Edge generation
            new_adj = self._generate_edges(new_embeddings, adj_matrix, num_nodes, num_to_generate)

            return new_embeddings, new_labels, new_adj
        else:
            return node_embeddings, labels, adj_matrix

    def _generate_edges(self, embeddings, old_adj, num_old_nodes, num_new_nodes):
        """
        Generate edges for new nodes
        Corresponds to Eq. (16)-(19)
        """
        device = embeddings.device
        total_nodes = num_old_nodes + num_new_nodes

        # Initialize new adjacency matrix
        new_adj = torch.zeros(total_nodes, total_nodes).to(device)
        new_adj[:num_old_nodes, :num_old_nodes] = old_adj

        # Edge generation for synthetic nodes
        for i in range(num_old_nodes, total_nodes):
            new_node_emb = embeddings[i:i + 1]

            # Eq. (16): E_{n,m} = σ(h_n'ᵀ S h_m)
            scores = torch.mm(torch.mm(new_node_emb, self.edge_decoder_weight),
                              embeddings[:num_old_nodes].t())
            scores = torch.sigmoid(scores).squeeze()

            # Eq. (18): Edge thresholding
            threshold = 0.5
            edges = (scores > threshold).float()

            new_adj[i, :num_old_nodes] = edges
            new_adj[:num_old_nodes, i] = edges  # Undirected graph

        return new_adj

    def compute_completion_loss(self, original_embeddings, reconstructed_embeddings, mask):
        """
        Attribute completion loss
        Eq. (13): L_completion
        """
        if mask.sum() == 0:
            return torch.tensor(0.0).to(original_embeddings.device)

        mse_loss = F.mse_loss(
            reconstructed_embeddings[mask],
            original_embeddings[mask],
            reduction='mean'
        )
        return mse_loss

    def compute_edge_loss(self, pred_adj, true_adj):
        """
        Edge reconstruction loss
        Eq. (17): L_edge
        """
        loss = F.binary_cross_entropy_with_logits(pred_adj, true_adj, reduction='mean')
        return loss

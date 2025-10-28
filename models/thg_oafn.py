"""
THG-OAFN Main Model
Integrates all modules to implement the full fraud detection framework
Based on Algorithm 1 and the overall architecture from the paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru_gnn import GRU_GNN
from .graph_smote import GraphSMOTE
from .attention import MultiLayerAttention


class THG_OAFN(nn.Module):
    """
    Temporal-aware Heterogeneous Graph Oversampling and Attention Fusion Network
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_relations=4,
                 num_heads=8, num_hops=2, dropout=0.6, oversample_ratio=2.0):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (embedding size)
            output_dim: Number of output classes (2: normal/fraud)
            num_relations: Number of relation types
            num_heads: Number of attention heads
            num_hops: Number of hops in GNN
            dropout: Dropout rate
            oversample_ratio: Oversampling ratio
        """
        super(THG_OAFN, self).__init__()
        self.hidden_dim = hidden_dim
        self.oversample_ratio = oversample_ratio

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU-GNN temporal module
        self.gru_gnn = GRU_GNN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2
        )

        # GraphSMOTE oversampling module
        self.graph_smote = GraphSMOTE(
            embedding_dim=hidden_dim,
            k_neighbors=5
        )

        # Multi-layer attention mechanism
        self.multi_attention = MultiLayerAttention(
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_hops=num_hops,
            num_heads=num_heads
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_features, labels=None, adj_matrices=None, training=True):
        """
        Forward propagation
        Corresponds to Algorithm 1 in the paper.

        Args:
            g: DGL graph
            node_features: Node features (N, input_dim)
            labels: Node labels (required for training)
            adj_matrices: List of adjacency matrices
            training: Training mode flag

        Returns:
            logits: Classification logits
            embeddings: Node embeddings
            losses: Loss dictionary (if training)
        """
        # Step 1: Initial feature extraction
        h_init = self.feature_extractor(node_features)

        # Step 2: GRU-GNN temporal modeling
        h_temporal = self.gru_gnn(g, h_init)

        # Step 3: Oversampling (training only)
        graph_changed = False
        if training and labels is not None and self.oversample_ratio > 1.0:
            num_fraud = (labels == 1).sum().item()
            if num_fraud > 0:
                h_oversampled, labels_oversampled, adj_oversampled = self.graph_smote(
                    h_temporal, labels,
                    adj_matrices[0] if adj_matrices else torch.eye(h_temporal.shape[0]).to(h_temporal.device),
                    oversample_ratio=self.oversample_ratio
                )

                if h_oversampled.shape[0] != h_temporal.shape[0]:
                    graph_changed = True
                    print(f"  Warning: Graph SMOTE changed node count from {h_temporal.shape[0]} to {h_oversampled.shape[0]}")
                    print(f"  Skipping graph-based layers for this batch")

                h_processed = h_oversampled
                labels_processed = labels_oversampled
            else:
                print(f"  Warning: No fraud samples to oversample, skipping Graph SMOTE")
                h_processed = h_temporal
                labels_processed = labels
        else:
            h_processed = h_temporal
            labels_processed = labels

        # Step 4: Multi-layer attention fusion
        if graph_changed:
            h_attention = h_processed
        else:
            if adj_matrices is None:
                num_nodes = h_processed.shape[0]
                adj_matrices = None
            h_attention = self.multi_attention(g, h_processed, adj_matrices)

        # Step 5: Classification
        h_final = self.dropout(h_attention)
        logits = self.classifier(h_final)

        # Compute loss (for training)
        losses = {}
        if training and labels_processed is not None:
            num_fraud = (labels_processed == 1).sum().float()
            num_normal = (labels_processed == 0).sum().float()

            if num_fraud > 0 and num_normal > 0:
                weight_ratio = (num_normal / num_fraud).sqrt()
                weight_fraud = torch.clamp(weight_ratio, min=1.5, max=5.0).item()
                class_weights = torch.tensor([1.0, weight_fraud]).to(logits.device)
                print(f"  Class weights: Normal=1.0, Fraud={weight_fraud:.2f} (samples: {int(num_normal)}/{int(num_fraud)})")
            else:
                class_weights = None

            loss_cls = F.cross_entropy(logits, labels_processed, weight=class_weights)
            losses['loss_cls'] = loss_cls
            losses['total_loss'] = loss_cls

        return logits, h_attention, losses if training else None

    def predict(self, g, node_features, adj_matrices=None):
        """
        Inference mode prediction

        Args:
            g: DGL graph
            node_features: Node features
            adj_matrices: List of adjacency matrices

        Returns:
            predictions: Predicted labels
            probabilities: Predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            logits, embeddings, _ = self.forward(
                g, node_features,
                labels=None,
                adj_matrices=adj_matrices,
                training=False
            )
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


def create_thg_oafn_model(config):
    """
    Create a THG-OAFN model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        model: THG-OAFN instance
    """
    model = THG_OAFN(
        input_dim=config.get('input_dim', 15),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 2),
        num_relations=config.get('num_relations', 4),
        num_heads=config.get('num_heads', 8),
        num_hops=config.get('num_hops', 2),
        dropout=config.get('dropout', 0.6),
        oversample_ratio=config.get('oversample_ratio', 2.0)
    )
    return model

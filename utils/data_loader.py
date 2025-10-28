"""
Data Loading and Preprocessing Module
Handles heterogeneous graph data construction.
"""
import torch
import dgl
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HeterogeneousGraphDataset:
    """Heterogeneous Graph Dataset Class"""

    def __init__(self, node_features, edge_index, edge_types, labels, timestamps=None):
        """
        Args:
            node_features: Node feature matrix (N, F)
            edge_index: Edge index (2, E)
            edge_types: Edge types (E,)
            labels: Node labels (N,)
            timestamps: Optional timestamps (N,)
        """
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.labels = labels
        self.timestamps = timestamps
        self.num_nodes = node_features.shape[0]
        self.num_edges = edge_index.shape[1]
        self.num_features = node_features.shape[1]
        self.num_classes = len(np.unique(labels))

    def build_dgl_graph(self, use_homogeneous=True):
        """
        Build a DGL graph.

        Args:
            use_homogeneous: Whether to use a homogeneous graph (recommended to avoid numpy.int64 issues)
        """
        if use_homogeneous:
            # Homogeneous graph, edge types stored as edge attributes
            src = torch.LongTensor(self.edge_index[0])
            dst = torch.LongTensor(self.edge_index[1])

            g = dgl.graph((src, dst), num_nodes=int(self.num_nodes))
            g.edata['edge_type'] = torch.LongTensor(self.edge_types)
            g.ndata['feat'] = torch.FloatTensor(self.node_features)
            g.ndata['label'] = torch.LongTensor(self.labels)

            if self.timestamps is not None:
                if len(self.timestamps) == self.num_nodes:
                    g.ndata['timestamp'] = torch.FloatTensor(self.timestamps)
                else:
                    print(f"  Warning: timestamp length ({len(self.timestamps)}) != nodes ({self.num_nodes}), skipping timestamps")

        else:
            # Heterogeneous graph (may cause numpy.int64 issues)
            unique_types = np.unique(self.edge_types)
            edge_dict = {}

            for etype in unique_types:
                mask = self.edge_types == etype
                src_indices = torch.LongTensor(self.edge_index[0, mask])
                dst_indices = torch.LongTensor(self.edge_index[1, mask])
                edge_dict[('node', f'etype_{int(etype)}', 'node')] = (src_indices, dst_indices)

            g = dgl.heterograph(edge_dict, num_nodes_dict={'node': int(self.num_nodes)})
            g.nodes['node'].data['feat'] = torch.FloatTensor(self.node_features)
            g.nodes['node'].data['label'] = torch.LongTensor(self.labels)

            if self.timestamps is not None:
                if len(self.timestamps) == self.num_nodes:
                    g.nodes['node'].data['timestamp'] = torch.FloatTensor(self.timestamps)
                else:
                    print(f"  Warning: timestamp length ({len(self.timestamps)}) != nodes ({self.num_nodes}), skipping timestamps")

        return g


def load_synthetic_data(num_nodes=1000, num_features=15, fraud_ratio=0.03):
    """
    Generate synthetic fraud detection data simulating Amazon/YelpChi.

    Args:
        num_nodes: Number of nodes
        num_features: Number of features
        fraud_ratio: Ratio of fraudulent nodes

    Returns:
        dataset: HeterogeneousGraphDataset object
    """
    np.random.seed(42)
    node_features = np.random.randn(num_nodes, num_features)

    num_fraud = int(num_nodes * fraud_ratio)
    labels = np.zeros(num_nodes, dtype=int)
    labels[:num_fraud] = 1
    np.random.shuffle(labels)

    fraud_mask = labels == 1
    node_features[fraud_mask] += np.random.randn(num_fraud, num_features) * 2

    num_edges = num_nodes * 5
    src_nodes = np.random.randint(0, num_nodes, num_edges)
    dst_nodes = np.random.randint(0, num_nodes, num_edges)
    mask = src_nodes != dst_nodes
    src_nodes = src_nodes[mask]
    dst_nodes = dst_nodes[mask]
    edge_index = np.vstack([src_nodes, dst_nodes])

    num_edge_types = 4
    edge_types = np.random.randint(0, num_edge_types, edge_index.shape[1])

    for i in range(edge_index.shape[1]):
        if labels[src_nodes[i]] == 1 or labels[dst_nodes[i]] == 1:
            if np.random.rand() > 0.5:
                edge_types[i] = np.random.choice([1, 2])  # refund or complaint

    timestamps = np.sort(np.random.rand(num_nodes))

    dataset = HeterogeneousGraphDataset(
        node_features=node_features,
        edge_index=edge_index,
        edge_types=edge_types,
        labels=labels,
        timestamps=timestamps
    )
    return dataset


def split_data(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split the dataset into train/val/test sets based on timestamps.

    Args:
        dataset: HeterogeneousGraphDataset object
        train_ratio: Train set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        train_mask, val_mask, test_mask: Boolean masks
    """
    num_nodes = dataset.num_nodes
    indices = np.arange(num_nodes)

    sorted_indices = np.argsort(dataset.timestamps) if dataset.timestamps is not None else indices
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_indices = sorted_indices[:train_size]
    val_indices = sorted_indices[train_size:train_size + val_size]
    test_indices = sorted_indices[train_size + val_size:]

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask


def normalize_features(features, scaler=None):
    """
    Normalize node features.

    Args:
        features: Feature matrix
        scaler: Optional StandardScaler

    Returns:
        normalized_features: Scaled features
        scaler: Used scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = scaler.transform(features)
    return normalized_features, scaler


def load_real_dataset(dataset_name, data_dir='./data'):
    """
    Load real-world fraud detection datasets (Amazon, YelpChi, etc.)

    Supported formats:
        1. .mat file — {dataset_name}.mat
        2. .tsv/.csv — {dataset_name}_nodes and {dataset_name}_edges
    """
    dataset_name_lower = dataset_name.lower()
    mat_file = os.path.join(data_dir, f'{dataset_name}.mat')
    if os.path.exists(mat_file):
        return load_mat_dataset(mat_file, dataset_name)

    possible_paths = [
        (os.path.join(data_dir, dataset_name_lower, f'{dataset_name_lower}_nodes.tsv'),
         os.path.join(data_dir, dataset_name_lower, f'{dataset_name_lower}_edges.tsv')),
        (os.path.join(data_dir, dataset_name_lower, f'{dataset_name_lower}_nodes.csv'),
         os.path.join(data_dir, dataset_name_lower, f'{dataset_name_lower}_edges.csv')),
        (os.path.join(data_dir, f'{dataset_name_lower}_nodes.tsv'),
         os.path.join(data_dir, f'{dataset_name_lower}_edges.tsv')),
        (os.path.join(data_dir, f'{dataset_name_lower}_nodes.csv'),
         os.path.join(data_dir, f'{dataset_name_lower}_edges.csv')),
    ]

    nodes_file, edges_file = None, None
    for nodes_path, edges_path in possible_paths:
        if os.path.exists(nodes_path) and os.path.exists(edges_path):
            nodes_file, edges_file = nodes_path, edges_path
            break

    if nodes_file is None or edges_file is None:
        raise FileNotFoundError(
            f"Dataset files not found.\nExpected one of:\n  - {mat_file}\n" +
            "\n".join([f"  - {n} and {e}" for n, e in possible_paths])
        )

    return load_tsv_csv_dataset(nodes_file, edges_file)


def load_mat_dataset(mat_file, dataset_name='amazon'):
    """Load MATLAB .mat dataset"""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("Please install scipy to load .mat files: pip install scipy")

    print(f"Loading {dataset_name} from {mat_file}...")
    mat = sio.loadmat(mat_file)
    features = mat['features'].toarray() if hasattr(mat['features'], 'toarray') else mat['features']
    labels = mat['label'].flatten()

    edge_index_list, edge_type_list = [], []
    if 'net_upu' in mat or 'net_usu' in mat or 'net_uvu' in mat:
        for etype, key in enumerate(['net_upu', 'net_usu', 'net_uvu']):
            if key in mat:
                net = mat[key].tocoo()
                edges = np.vstack([net.row, net.col])
                edge_index_list.append(edges)
                edge_type_list.append(np.full(edges.shape[1], etype, dtype=np.int64))

        edge_index = np.hstack(edge_index_list)
        edge_types = np.concatenate(edge_type_list)
    elif 'homo' in mat:
        adj_matrix = mat['homo'].tocoo()
        edge_index = np.vstack([adj_matrix.row, adj_matrix.col])
        edge_types = np.zeros(edge_index.shape[1], dtype=np.int64)
    else:
        raise ValueError(f"No graph structure ('homo' or 'net_*') found in {mat_file}")

    features, _ = normalize_features(features)
    num_nodes = len(labels)

    print(f"Dataset loaded: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"Features: {features.shape[1]} dimensions")
    print(f"Labels: {len(np.unique(labels))} classes")
    print(f"Fraud ratio: {(labels == 1).sum() / len(labels):.2%}")

    dataset = HeterogeneousGraphDataset(features, edge_index, edge_types, labels)
    return dataset


def load_tsv_csv_dataset(nodes_file, edges_file):
    """Load TSV/CSV dataset"""
    separator = '\t' if nodes_file.endswith('.tsv') else ','
    print(f"Loading nodes from {nodes_file}...")
    nodes_df = pd.read_csv(nodes_file, sep=separator)

    if 'label' not in nodes_df.columns:
        raise ValueError("Node file must contain a 'label' column")

    labels = nodes_df['label'].values.astype(np.int64)
    feature_cols = [c for c in nodes_df.columns if c not in ['node_id', 'label']]
    node_features = nodes_df[feature_cols].values.astype(np.float32)
    node_features, _ = normalize_features(node_features)

    print(f"Loading edges from {edges_file}...")
    edges_df = pd.read_csv(edges_file, sep=separator)
    src, dst = edges_df['src'].values, edges_df['dst'].values
    edge_index = np.vstack([src, dst]).astype(np.int64)

    edge_types = edges_df['edge_type'].values.astype(np.int64) if 'edge_type' in edges_df.columns else np.zeros(len(edges_df), dtype=np.int64)
    timestamps = None
    if 'timestamp' in nodes_df.columns:
        timestamps = nodes_df['timestamp'].values
        timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)

    print(f"Dataset loaded: {len(labels)} nodes, {edge_index.shape[1]} edges")
    print(f"Features: {node_features.shape[1]} dimensions")
    print(f"Fraud ratio: {(labels == 1).sum() / len(labels):.2%}")

    dataset = HeterogeneousGraphDataset(node_features, edge_index, edge_types, labels, timestamps)
    return dataset

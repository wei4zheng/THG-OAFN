"""
THG-OAFN Training Script
Implements the complete training, validation, and testing pipeline.
"""
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.thg_oafn import create_thg_oafn_model
from utils.data_loader import load_synthetic_data, load_real_dataset, split_data
from utils.metrics import calculate_metrics, print_metrics


def train_epoch(model, g, features, labels, train_mask, optimizer, device):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()

    # Move data to device
    features = features.to(device)
    labels = labels.to(device)
    g = g.to(device)

    # Forward propagation
    logits, embeddings, losses = model(
        g, features, labels, training=True
    )

    # Use model-internal loss (already considers oversampling)
    loss = losses['total_loss']

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()




def evaluate(model, g, features, labels, mask, device):
    """Evaluate the model"""
    model.eval()

    features = features.to(device)
    labels = labels.to(device)
    g = g.to(device)

    with torch.no_grad():
        predictions, probabilities = model.predict(g, features)

    # Evaluate only on the specified mask
    y_true = labels[mask].cpu().numpy()
    y_pred = predictions[mask].cpu().numpy()
    y_prob = probabilities[mask, 1].cpu().numpy()  # Probability of fraud class

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    return metrics


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    if args.dataset:
        # Load real dataset
        dataset = load_real_dataset(args.dataset, data_dir='./data')
        # Update configuration to match real dataset dimensions
        args.num_features = dataset.num_features
        args.num_relations = len(np.unique(dataset.edge_types))
    else:
        # Load synthetic data
        dataset = load_synthetic_data(
            num_nodes=args.num_nodes,
            num_features=args.num_features,
            fraud_ratio=args.fraud_ratio
        )

    # Build DGL graph
    g = dataset.build_dgl_graph()
    features = torch.FloatTensor(dataset.node_features)
    labels = torch.LongTensor(dataset.labels)

    # Split dataset
    train_mask, val_mask, test_mask = split_data(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    print(f"Dataset stats:")
    print(f"  Total nodes: {dataset.num_nodes}")
    print(f"  Train nodes: {train_mask.sum()}")
    print(f"  Val nodes: {val_mask.sum()}")
    print(f"  Test nodes: {test_mask.sum()}")
    print(f"  Fraud ratio: {(labels == 1).sum().item() / len(labels):.2%}")

    # Verify temporal split (if timestamps exist)
    if dataset.timestamps is not None:
        import datetime
        train_timestamps = dataset.timestamps[train_mask]
        val_timestamps = dataset.timestamps[val_mask]
        test_timestamps = dataset.timestamps[test_mask]

        def ts_to_date(ts):
            """Convert timestamp to date string"""
            return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

        print(f"\nTemporal split verification:")
        print(f"  Train: {ts_to_date(train_timestamps.min())} to {ts_to_date(train_timestamps.max())}")
        print(f"  Val:   {ts_to_date(val_timestamps.min())} to {ts_to_date(val_timestamps.max())}")
        print(f"  Test:  {ts_to_date(test_timestamps.min())} to {ts_to_date(test_timestamps.max())}")
        print(f"  Train fraud ratio: {(labels[train_mask] == 1).sum().item() / train_mask.sum():.2%}")
        print(f"  Val fraud ratio:   {(labels[val_mask] == 1).sum().item() / val_mask.sum():.2%}")
        print(f"  Test fraud ratio:  {(labels[test_mask] == 1).sum().item() / test_mask.sum():.2%}")

    # Create model
    print("\nCreating model...")
    config = {
        'input_dim': args.num_features,
        'hidden_dim': args.embedding_dim,
        'output_dim': 2,
        'num_relations': args.num_relations,
        'num_heads': args.num_heads,
        'num_hops': args.num_hops,
        'dropout': args.dropout,
        'oversample_ratio': args.oversample_ratio
    }

    model = create_thg_oafn_model(config).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(
            model, g, features, labels,
            train_mask, optimizer, device
        )

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, g, features, labels, val_mask, device)

            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUC: {val_metrics.get('AUC', 0):.4f}")
            print(f"  Val F1: {val_metrics.get('F1', 0):.4f}")
            print(f"  Val Recall: {val_metrics.get('Recall', 0):.4f}")

            # Early stopping
            if val_metrics['F1'] > best_val_f1:
                best_val_f1 = val_metrics['F1']
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate(model, g, features, labels, test_mask, device)

    print_metrics(test_metrics, "Test")

    # Save results
    results = {
        'test_auc': test_metrics.get('AUC', 0),
        'test_f1': test_metrics.get('F1', 0),
        'test_recall': test_metrics.get('Recall', 0),
        'test_precision': test_metrics.get('Precision', 0),
        'config': config
    }

    torch.save(results, 'results.pth')
    print("\nResults saved to results.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train THG-OAFN model')

    # Data parameters
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (amazon, yelpchi, etc.). If not specified, use synthetic data')
    parser.add_argument('--num_nodes', type=int, default=1000, help='Number of nodes (for synthetic data)')
    parser.add_argument('--num_features', type=int, default=15, help='Number of features (for synthetic data)')
    parser.add_argument('--fraud_ratio', type=float, default=0.03, help='Fraud ratio (for synthetic data)')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Training ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')

    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_relations', type=int, default=4, help='Number of relations')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_hops', type=int, default=2, help='Number of hops')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--oversample_ratio', type=float, default=2.0, help='Oversample ratio')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    main(args)

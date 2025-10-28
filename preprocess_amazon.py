"""
Amazon Review Data Preprocessing Script
Convert raw Amazon review data into a graph structure (nodes and edges)
"""
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import argparse


def load_amazon_reviews(data_dir, max_files=None, sample_size=None):
    """
    Load Amazon review data

    Args:
        data_dir: Data directory
        max_files: Maximum number of files to load (None = load all)
        sample_size: Number of rows to sample per file (None = load all)

    Returns:
        reviews_df: Combined review DataFrame
    """
    # Find all tsv files
    files = [f for f in os.listdir(data_dir) if f.startswith('amazon_reviews_us_') and f.endswith('.tsv')]

    if max_files:
        files = files[:max_files]

    print(f"Found {len(files)} Amazon review files")

    all_reviews = []

    for file in files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file}...")

        try:
            # Standard columns in Amazon review datasets
            df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', low_memory=False)

            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)

            all_reviews.append(df)
            print(f"  Loaded {len(df)} reviews")

        except Exception as e:
            print(f"  Error loading {file}: {e}")
            continue

    if not all_reviews:
        raise ValueError("No data files were successfully loaded")

    # Combine all reviews
    reviews_df = pd.concat(all_reviews, ignore_index=True)
    print(f"\nTotal reviews loaded: {len(reviews_df)}")

    return reviews_df


def build_graph_from_reviews(reviews_df, fraud_detection_mode='rating_based'):
    """
    Build a graph structure from review data

    Args:
        reviews_df: Review DataFrame
        fraud_detection_mode: Fraud detection mode
            - 'rating_based': Detect fraud based on ratings (extreme scores, frequent reviews, etc.)
            - 'verified_purchase': Based on verified purchase status

    Returns:
        nodes_df: Node data (users and products)
        edges_df: Edge data (user–product relationships)
    """
    print("\nBuilding graph structure...")

    # Required columns
    required_cols = ['customer_id', 'product_id', 'star_rating']
    for col in required_cols:
        if col not in reviews_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Clean data
    reviews_df = reviews_df.dropna(subset=required_cols)

    # Build user nodes
    print("Building user nodes...")
    user_features = defaultdict(lambda: {
        'review_count': 0,
        'avg_rating': 0,
        'ratings': [],
        'timestamps': [],
        'low_rating_count': 0  # Count of 1–2 star ratings (proxy for refund rate)
    })

    for _, row in reviews_df.iterrows():
        user_id = row['customer_id']
        rating = float(row['star_rating'])

        user_features[user_id]['review_count'] += 1
        user_features[user_id]['ratings'].append(rating)

        # Count low ratings (proxy for refund behavior)
        if rating <= 2.0:
            user_features[user_id]['low_rating_count'] += 1

        # Record timestamp (for temporal split and frequency)
        if 'review_date' in reviews_df.columns:
            try:
                timestamp = pd.to_datetime(row['review_date']).timestamp()
                user_features[user_id]['timestamps'].append(timestamp)
            except:
                pass

        # Handle verified_purchase column
        if 'verified_purchase' in reviews_df.columns:
            if 'verified_count' not in user_features[user_id]:
                user_features[user_id]['verified_count'] = 0
            if row['verified_purchase'] == 'Y':
                user_features[user_id]['verified_count'] += 1

        # Handle helpful_votes column
        if 'helpful_votes' in reviews_df.columns:
            if 'total_helpful' not in user_features[user_id]:
                user_features[user_id]['total_helpful'] = 0
            user_features[user_id]['total_helpful'] += float(row.get('helpful_votes', 0))

    # Compute user features
    user_nodes = []
    for user_id, features in user_features.items():
        ratings = features['ratings']

        node = {
            'node_id': f'user_{user_id}',
            'node_type': 'user',
            'review_count': features['review_count'],
            'avg_rating': np.mean(ratings),
            'rating_std': np.std(ratings),
            'rating_min': np.min(ratings),
            'rating_max': np.max(ratings),
        }

        # Add timestamp (last review time)
        if features['timestamps']:
            node['timestamp'] = max(features['timestamps'])

        # Add extra features
        if 'verified_count' in features:
            node['verified_ratio'] = features['verified_count'] / features['review_count']

        if 'total_helpful' in features:
            node['avg_helpful'] = features['total_helpful'] / features['review_count']

        # Fraud labeling (heuristic rules)
        # Paper reports Amazon fraud rate ≈ 3.2%, adjust rules to match similar ratio
        is_fraud = 0
        if fraud_detection_mode == 'rating_based':
            # Method 1: Extreme ratings + low variance + high frequency (strict triple condition)
            extreme_rating = (node['avg_rating'] <= 1.8 or node['avg_rating'] >= 4.7)
            low_variance = node['rating_std'] < 0.5
            high_frequency = node['review_count'] > 15

            # Method 2: Very extreme rating + moderately high frequency
            very_extreme_and_frequent = (
                (node['avg_rating'] <= 1.3 or node['avg_rating'] >= 4.9) and
                node['review_count'] > 10
            )

            # Method 3: Very low variance + high frequency (uniform bot-like pattern)
            bot_like_pattern = (
                node['rating_std'] < 0.2 and
                node['review_count'] > 20
            )

            # Mark as fraud if any condition is met
            if (extreme_rating and low_variance and high_frequency) or very_extreme_and_frequent or bot_like_pattern:
                is_fraud = 1

        elif fraud_detection_mode == 'verified_purchase' and 'verified_ratio' in node:
            # Relaxed rule for verified purchase ratio
            if node['verified_ratio'] < 0.5 and node['review_count'] > 10:  # from 0.3 & 20 to 0.5 & 10
                is_fraud = 1

        node['label'] = is_fraud
        user_nodes.append(node)

    # Build product nodes
    print("Building product nodes...")
    product_features = defaultdict(lambda: {'review_count': 0, 'ratings': [], 'timestamps': []})

    for _, row in reviews_df.iterrows():
        product_id = row['product_id']
        rating = float(row['star_rating'])

        product_features[product_id]['review_count'] += 1
        product_features[product_id]['ratings'].append(rating)

        # Record timestamp (for temporal split)
        if 'review_date' in reviews_df.columns:
            try:
                timestamp = pd.to_datetime(row['review_date']).timestamp()
                product_features[product_id]['timestamps'].append(timestamp)
            except:
                pass

    product_nodes = []
    for product_id, features in product_features.items():
        ratings = features['ratings']

        node = {
            'node_id': f'product_{product_id}',
            'node_type': 'product',
            'review_count': features['review_count'],
            'avg_rating': np.mean(ratings),
            'rating_std': np.std(ratings),
            'rating_min': np.min(ratings),
            'rating_max': np.max(ratings),
            'label': 0  # Product nodes are non-fraudulent by default
        }

        # Add timestamp (last review time)
        if features['timestamps']:
            node['timestamp'] = max(features['timestamps'])

        product_nodes.append(node)

    # Merge nodes
    nodes_df = pd.DataFrame(user_nodes + product_nodes)

    # Re-map node IDs to contiguous integers
    node_id_map = {node_id: idx for idx, node_id in enumerate(nodes_df['node_id'])}
    nodes_df['original_id'] = nodes_df['node_id']
    nodes_df['node_id'] = nodes_df['node_id'].map(node_id_map)

    print(f"Created {len(user_nodes)} user nodes and {len(product_nodes)} product nodes")
    print(f"Fraud ratio: {nodes_df['label'].sum() / len(nodes_df):.2%}")

    # Build edges
    print("Building edges...")
    edges = []

    for _, row in reviews_df.iterrows():
        user_id = f'user_{row["customer_id"]}'
        product_id = f'product_{row["product_id"]}'

        if user_id not in node_id_map or product_id not in node_id_map:
            continue

        edge = {
            'src': int(node_id_map[user_id]),
            'dst': int(node_id_map[product_id]),
            'edge_type': 0,  # Review relationship
            'rating': float(row['star_rating'])
        }

        # Add timestamp (if available)
        if 'review_date' in reviews_df.columns:
            try:
                edge['timestamp'] = pd.to_datetime(row['review_date']).timestamp()
            except:
                pass

        edges.append(edge)

    edges_df = pd.DataFrame(edges)

    print(f"Created {len(edges_df)} edges")

    return nodes_df, edges_df


def save_graph_data(nodes_df, edges_df, output_dir, dataset_name='amazon'):
    """Save graph data"""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare node data (keep only features and labels)
    feature_cols = [col for col in nodes_df.columns if col not in ['node_id', 'label', 'original_id', 'node_type']]
    nodes_output = nodes_df[['node_id', 'label'] + feature_cols].copy()

    # Ensure correct data types
    nodes_output['node_id'] = nodes_output['node_id'].astype(int)
    nodes_output['label'] = nodes_output['label'].astype(int)

    # Ensure correct edge data types
    edges_output = edges_df.copy()
    edges_output['src'] = edges_output['src'].astype(int)
    edges_output['dst'] = edges_output['dst'].astype(int)
    edges_output['edge_type'] = edges_output['edge_type'].astype(int)

    # Save
    nodes_file = os.path.join(output_dir, f'{dataset_name}_nodes.tsv')
    edges_file = os.path.join(output_dir, f'{dataset_name}_edges.tsv')

    nodes_output.to_csv(nodes_file, sep='\t', index=False)
    edges_output.to_csv(edges_file, sep='\t', index=False)

    print(f"\nSaved graph data:")
    print(f"  Nodes: {nodes_file}")
    print(f"  Edges: {edges_file}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Amazon reviews data')
    parser.add_argument('--data_dir', type=str, default='./data/amazon',
                        help='Directory containing Amazon review TSV files')
    parser.add_argument('--output_dir', type=str, default='./data/amazon',
                        help='Output directory for processed graph data')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (None for all)')
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='Sample size per file (None for all rows)')
    parser.add_argument('--fraud_mode', type=str, default='rating_based',
                        choices=['rating_based', 'verified_purchase'],
                        help='Fraud detection mode')

    args = parser.parse_args()

    # Load data
    reviews_df = load_amazon_reviews(args.data_dir, args.max_files, args.sample_size)

    # Build graph
    nodes_df, edges_df = build_graph_from_reviews(reviews_df, args.fraud_mode)

    # Save
    save_graph_data(nodes_df, edges_df, args.output_dir)

    print("\nPreprocessing complete!")
    print(f"You can now train the model with: python train.py --dataset amazon --use_gpu")


if __name__ == '__main__':
    main()

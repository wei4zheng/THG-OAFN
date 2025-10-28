# THG-OAFN: Temporal-aware Heterogeneous Graph Oversampling and Attention Fusion Network

PyTorch implementation based on the paper “Internet Fraud Transaction Detection based on Temporal-aware Heterogeneous Graph Oversampling and Attention Fusion Network.”

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DGL 1.1+](https://img.shields.io/badge/dgl-1.1+-orange.svg)](https://www.dgl.ai/)

---

## 📋 Table of Contents

- [Model Overview](#model-overview)
- [Environment Requirements](#environment-requirements)
- [Installation Steps](#installation-steps)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Detailed Usage Instructions](#detailed-usage-instructions)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [Experimental Results](#experimental-results)
- [Citation](#citation)


---

## 🎯 Model Overview

THG-OAFN is a deep learning framework for internet fraud transaction detection. Its main innovations include the following:

### Key Features

1. **Temporal-aware Heterogeneous Graph Modeling**
   - Abstracts transaction data into a heterogeneous graph structure  
   - Utilizes a GRU-GNN fusion module to capture temporal dynamics  
   - Supports multiple relationship types (e.g., user–product, user–rating, user–review)

2. **GraphSMOTE-based Graph Oversampling**
   - An improved graph oversampling technique to address class imbalance  
   - Maintains structural integrity of fraud clusters through k-hop topological constraints  
   - Automatically generates synthetic fraud samples  

3. **Multi-level Attention Mechanism**
   - *Relation Fusion Layer:* Integrates information from different relation types  
   - *Neighborhood Fusion Layer:* Aggregates features from neighboring nodes  
   - *Information-aware Layer:* Learns importance weights of key features  

4. **Class Weight Balancing**
   - Dynamically computes adaptive class weights  
   - Applies square-root scaling to mitigate extreme imbalance  
   - Restricts weight values within the range [1.5, 5.0]

---

## 💻 Environment Requirements

### Hardware Requirements

- **CPU:** Intel i5 or higher recommended  
- **Memory:** Minimum 8 GB RAM (16 GB+ recommended)  
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ (recommended for training)  
  - **VRAM:** Minimum 4 GB (8 GB+ recommended)

### Software Requirements

- **Operating System:** Windows 10/11, Linux (Ubuntu 18.04+), or macOS  
- **Python:** Version 3.8 or higher  
- **CUDA** (optional): Version 11.8 or 12.1 (if using GPU)

---

## 🚀 Installation Steps

### Method 1: CPU Version (Quick Testing)


```bash
# 1. Clone or download the project
cd /path/to/project

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```

### Method 2: GPU Version (Recommended for Training)


```bash
# 1-2. Same as above (create and activate the environment)

# 3. Install basic dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm

# 4. Install the PyTorch GPU version
# Visit https://pytorch.org to select the appropriate CUDA version
# Example (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install the DGL GPU version
# CUDA 11.8:
pip install dgl-cu118 -f https://data.dgl.ai/wheels/repo.html
# CUDA 12.1:
pip install dgl-cu121 -f https://data.dgl.ai/wheels/repo.html

```

### Verify Installation


```bash
python -c "import torch; import dgl; print(f'PyTorch: {torch.__version__}'); print(f'DGL: {dgl.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected Output:

```
PyTorch: 2.x.x
DGL: 1.x.x
CUDA available: True  # If use GPU
```

---

## 📊 Dataset Preparation

### Supported Data Formats

#### Format 1: MATLAB .mat File (Recommended)

The project supports standard heterogeneous graph datasets in `.mat` format, which should include the following fields:


```
Amazon.mat  
├── features: Node feature matrix (N × F)  
├── label: Node labels (1 × N)  
├── homo: Homogeneous graph adjacency matrix (N × N, sparse)  
└── net_*: Heterogeneous relation networks (optional)  
  ├── net_upu: User–Product–User  
  ├── net_usu: User–Rating–User  
  └── net_uvu: User–Review–User  

```

**Dataset Location:**
```
project/
└── data/
    └── Amazon/
        └── Amazon.mat
```
or：
```
project/
└── data/
    └── Amazon.mat
```

#### Format 2: CSV/TSV Files

If you are using a custom dataset, two files are required:

**nodes.tsv/csv** (Node file):

```
node_id	label	feature_1	feature_2	...	feature_n	timestamp (optional)  
0	0	1.23	4.56	...	7.89	1609459200  
1	1	2.34	5.67	...	8.90	1609545600  
...  
``

```

**edges.tsv/csv**（Edge file）：
```
src	dst	edge_type	timestamp(optional)
0	1	0	1609459200
1	2	1	1609545600
...
```

### Obtain the Dataset


**Amazon Dataset**:  
- Source: Amazon Public Dataset  
- Scale: 11,944 nodes, ~9.5M edges, 25-dimensional features  
- Fraud Rate: 6.87%  


**YelpChi Dataset**:  
- Source: Yelp Open Dataset  
- Used for business review fraud detection  


---

## ⚡ Quick Start

### The Simplest Way to Run


```bash
# 1. Make sure the dataset is in the correct location
ls data/Amazon/Amazon.mat

# 2. Run training (CPU)
python train.py --dataset Amazon --data_dir ./data/Amazon --epochs 100

# 3. Run training (GPU)
python train.py --dataset Amazon --data_dir ./data/Amazon --epochs 100 --use_gpu

```

### Recommended Configuration (Balanced Performance and Accuracy)
`


```bash
python train.py \
    --dataset Amazon \
    --data_dir ./data/Amazon \
    --epochs 100 \
    --batch_size 256 \
    --embedding_dim 64 \
    --num_heads 4 \
    --num_hops 2 \
    --dropout 0.5 \
    --lr 0.01 \
    --weight_decay 0.0005 \
    --oversample_ratio 1.0 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --use_gpu
```

---
## 📖 Detailed Usage Instructions

### Training Arguments Explained

#### Dataset Arguments

| Argument        | Default     | Description                                           |
|-----------------|-------------|-------------------------------------------------------|
| `--dataset`     | `synthetic` | Dataset name (e.g., Amazon / YelpChi)                |
| `--data_dir`    | `./data`    | Path to the dataset directory                         |
| `--train_ratio` | `0.6`       | Training split ratio (60%)                            |
| `--val_ratio`   | `0.2`       | Validation split ratio (20%); remaining 20% is test   |

#### Model Arguments

| Argument           | Default | Description                      | Tuning Tips                         |
|--------------------|---------|----------------------------------|-------------------------------------|
| `--embedding_dim`  | `64`    | Embedding dimension              | 32/64/128; larger = slower          |
| `--num_heads`      | `8`     | Number of attention heads        | 4/8/16; affects model complexity    |
| `--num_hops`       | `2`     | Number of GNN layers (hops)      | 2–3; too deep may overfit           |
| `--dropout`        | `0.6`   | Dropout rate                     | 0.3–0.6 to prevent overfitting      |
| `--num_relations`  | `4`     | Number of relation types         | Adjust to match the dataset         |

#### Training Arguments

| Argument              | Default | Description          | Tuning Tips                              |
|-----------------------|---------|----------------------|------------------------------------------|
| `--epochs`            | `100`   | Number of epochs     | 50–200                                   |
| `--batch_size`        | `256`   | Batch size           | 128/256/512                              |
| `--lr`                | `0.01`  | Learning rate        | **Key hyperparameter**; 0.001–0.01       |
| `--weight_decay`      | `5e-4`  | L2 regularization    | 1e-4 to 1e-3                             |
| `--oversample_ratio`  | `2.0`   | Oversampling ratio   | Use **1.0 to disable**                   |
| `--patience`          | `20`    | Early stopping rounds| 10–30                                    |

#### Other Arguments

| Argument        | Default           | Description                |
|-----------------|-------------------|----------------------------|
| `--use_gpu`     | `False`           | Whether to use the GPU     |
| `--seed`        | `42`              | Random seed                |
| `--save_model`  | `True`            | Whether to save the model  |
| `--model_path`  | `./checkpoints`   | Path to save checkpoints   |

### Interpreting Training Outputs


```
Epoch 50/100
  Train Loss: 0.1234
  Val AUC: 0.8567
  Val F1: 0.7234
  Val Recall: 0.8012

Best epoch: 45
Test Results:
  AUC: 0.8654
  Precision: 0.7123
  Recall: 0.8234
  F1: 0.7634
  Accuracy: 0.9012
```

**Key Metrics**:  
- **AUC**: Normal range is 0.75–0.95; higher is better  
- **Recall**: Detection rate — very important! Should be between 70%–95%  
- **Precision**: Accuracy rate; should be between 50%–80%  
- **F1**: Harmonic mean of Precision and Recall; should be between 60%–85%  

### Common Command Examples

#### 1. Quick Test (Small Dataset, CPU)

```bash
python train.py --dataset Amazon --data_dir ./data/Amazon --epochs 20
```

#### 2. Standard Training (GPU, Oversampling Disabled)

```bash
python train.py \
    --dataset Amazon \
    --data_dir ./data/Amazon \
    --epochs 100 \
    --oversample_ratio 1.0 \
    --use_gpu
```

#### 3. Debug Mode (Verbose Logging)

```bash
python train.py \
    --dataset Amazon \
    --data_dir ./data/Amazon \
    --epochs 10 \
    --batch_size 64 \
    --use_gpu
```
#### 4. Lower Learning Rate (When Loss Becomes NaN)

```bash
python train.py \
    --dataset Amazon \
    --data_dir ./data/Amazon \
    --lr 0.001 \
    --use_gpu
```

#### 5. Adjust Oversampling (When F1 Score Is Too Low)

```bash
python train.py \
    --dataset Amazon \
    --data_dir ./data/Amazon \
    --oversample_ratio 1.5 \
    --use_gpu
```

---

## 📁 Project Struct

```
THG-OAFN/
├── data/                          # Data directory
│   ├── Amazon/
│   │   └── Amazon.mat            # Amazon dataset
│   └── synthetic/                 # Synthetic data (for testing)
│
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── thg_oafn.py               # Main model (THG-OAFN)
│   ├── gru_gnn.py                # GRU-GNN fusion module
│   ├── graph_smote.py            # GraphSMOTE oversampling
│   └── attention.py              # Multi-layer attention mechanism
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── data_loader.py            # Data loader (supports .mat and CSV)
│   └── metrics.py                # Evaluation metrics (AUC, F1, Recall, etc.)
│
├── checkpoints/                   # Model checkpoints (generated after training)
│
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── OVERSAMPLING_FIX.md           # Oversampling issue fix notes
├── MEMORY_FIX.md                 # Memory optimization notes
└── preprocess_amazon.py          # Amazon data preprocessing (optional)

```

---

## ❓ FAQ

### Q1: Loss becomes NaN during training

**Cause:** Learning rate too high or gradient explosion.

**Solutions:**
```bash
# Method 1: Lower the learning rate
python train.py --dataset Amazon --data_dir ./data/Amazon --lr 0.001 --use_gpu

# Method 2: Reduce the oversampling ratio
python train.py --dataset Amazon --data_dir ./data/Amazon --oversample_ratio 1.0 --use_gpu
```

### Q2: Both F1 and Recall are 0

**Cause:** The model predicts all samples as normal class.

**Solutions:**
- ✅ Already fixed in the code (added class weights)
- If the issue persists, disable oversampling: `--oversample_ratio 1.0`

### Q3: "Graph SMOTE changed node count" warning

**Impact:** The number of nodes in training and evaluation becomes inconsistent.

**Solution:**
```bash
# Disable Graph SMOTE
python train.py --dataset Amazon --data_dir ./data/Amazon --oversample_ratio 1.0 --use_gpu
```

### Q4: CUDA out of memory

**Cause:** Insufficient GPU memory.

**Solutions:**
```bash
# Method 1: Reduce batch size
python train.py --dataset Amazon --data_dir ./data/Amazon --batch_size 128 --use_gpu

# Method 2: Use CPU training
python train.py --dataset Amazon --data_dir ./data/Amazon
```

### Q5: Data file not found

**Error Message:** `FileNotFoundError: Data file not found`

**Solutions:**
```bash
# Check the file path
ls data/Amazon/Amazon.mat

# If the file is stored elsewhere, specify the correct path
python train.py --dataset Amazon --data_dir /path/to/your/data
```

### Q6: Precision is low but Recall is high

**Cause:** The model over-predicts the fraud class.

**Solutions:**
- Already fixed through class weight adjustment in the code.
- If the issue persists, adjust the weight range in `models/thg_oafn.py` (lines 163–165).

### Q7: Training is slow

**Optimization Tips:**
1. Use GPU: `--use_gpu`
2. Reduce batch size: `--batch_size 128`
3. Decrease the number of GNN layers: `--num_hops 2`
4. Disable oversampling: `--oversample_ratio 1.0`

---

## 📈 Experimental Results

### Amazon Dataset

| Metric | Paper Result | Expected in This Implementation |
|---------|---------------|--------------------------------|
| **AUC** | 96.56%        | 75–90%                         |
| **Recall** | 95.21%     | 70–85%                         |
| **Precision** | -        | 50–75%                         |
| **F1-score** | 94.72%   | 60–80%                         |

**Note:** Actual results depend on:
- Dataset version and preprocessing method  
- Hyperparameter configuration  
- Random seed  
- Hardware environment  

### Training Time (Reference)

| Hardware | Training Time (100 epochs) |
|-----------|----------------------------|
| CPU (Intel i7) | ~60–90 minutes |
| GPU (NVIDIA RTX 3060) | ~15–25 minutes |
| GPU (NVIDIA V100) | ~8–15 minutes |

---

## 🔧 Advanced Usage

### Custom Dataset

1. Prepare your data files (see [Dataset Preparation](#dataset-preparation)).  
2. Modify `utils/data_loader.py` to add custom loading logic.  
3. Run training.

### Model Tuning Recommendations

**Scenario 1: Low Recall (many false negatives)**  
- Increase the upper limit of fraud class weights (edit code).  
- Enable oversampling: `--oversample_ratio 1.5`  
- Lower classification threshold (edit code).  

**Scenario 2: Low Precision (many false positives)**  
- Decrease fraud class weights.  
- Disable oversampling: `--oversample_ratio 1.0`  
- Increase regularization: `--weight_decay 0.001`.  

**Scenario 3: Overfitting (good training performance, poor testing performance)**  
- Increase Dropout: `--dropout 0.6`  
- Add regularization: `--weight_decay 0.001`  
- Reduce model complexity: `--num_hops 2 --num_heads 4`.  

---

## 📚 Citation

If this project contributes to your research, please cite the original paper:

```bibtex
@article{thg_oafn_2024,
  title={Internet Fraud Transaction Detection based on Temporal-aware Heterogeneous Graph Oversampling and Attention Fusion Network},
  author={[Author Name]},
  journal={[Journal Name]},
  year={2024}
}
```

---

## 📄 License

This project is intended for academic research and educational purposes only.

---

## 🤝 Contribution & Feedback

For issues or suggestions, please submit an Issue or Pull Request.

### Known Issues

- [ ] Graph SMOTE may cause mismatched node counts  
- [ ] `.mat` files may not contain timestamp information  
- [ ] Large-scale datasets may encounter memory bottlenecks  

### TODO

- [ ] Add model inference script  
- [ ] Support distributed training  
- [ ] Add visualization tools  
- [ ] Support more dataset formats  

---

## 📞 Contact

- Repository: https://github.com/wei4zheng/THG-OAFN  
- Email: wei4zheng@xzit.edu.cn  

---

**Last Updated:** 2025-10-28

# HAM10000 Multimodal Skin Lesion Classification

AI-powered skin disease classification using **multimodal deep learning**.

- **Input**: Image + Age + Sex + Localization  
- **Output**: Diagnosis (7 classes: akiec, bcc, bkl, df, mel, nv, vasc)

## Features

- **Multimodal Architecture**: Combines image features with patient metadata
- **PyTorch + EfficientNetV2-S**: State-of-the-art backbone with mixed precision training
- **Web Application**: Modern, animated UI for predictions
- **No Data Leakage**: Lesion-based train/val split
- **GPU Optimized**: CUDA support with automatic mixed precision

## Dataset

This project uses the **HAM10000** ("Human Against Machine with 10000 training images") dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

### Disease Classes

| Code | Disease | Description |
|------|---------|-------------|
| akiec | Actinic Keratosis | Pre-cancerous scaly patches |
| bcc | Basal Cell Carcinoma | Most common skin cancer |
| bkl | Benign Keratosis | Non-cancerous skin growths |
| df | Dermatofibroma | Benign fibrous skin nodules |
| mel | Melanoma | Dangerous form of skin cancer |
| nv | Melanocytic Nevus | Common moles |
| vasc | Vascular Lesions | Blood vessel-related skin marks |

### Dataset Statistics
- **10,015** dermatoscopic images
- **7** diagnostic categories
- Includes patient metadata (age, sex, localization)

## Project Structure

```
skAIn/
├── dataset_train/
│   ├── image/              # Training images (.jpg or .png)
│   └── text/               # Training CSV file
├── dataset_test/
│   ├── image/              # Test images
│   └── text/               # Test CSV file
├── outputs/                # Model outputs (auto-created)
│   ├── best_model.pt       # Saved PyTorch model
│   ├── label_map.json      # Label mapping
│   ├── vocabularies.json   # Sex/localization encodings
│   ├── age_stats.json      # Age normalization stats
│   └── train_history.csv   # Training history
├── templates/              # Web app HTML templates
├── static/                 # Web app CSS and JS
├── train_multimodal_pytorch.py   # Training script (PyTorch)
├── app.py                  # Web application (Flask)
├── predict_test.py         # Batch prediction script
└── README.md               # This file
```

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm tqdm pandas scikit-learn pillow flask flask-cors

# 3. Verify GPU (optional)
python test_gpu_pytorch.py
```

## Usage

### 1. Training

```bash
python train_multimodal_pytorch.py
```

This will:
- Find the largest CSV in `dataset_train/text/`
- Split data by lesion_id (no leakage)
- Train EfficientNetV2-S backbone model
- Save best model to `outputs/best_model.pt`
- Log training history and metrics

### 2. Web Application

```bash
python app.py
```

Then open: **http://localhost:5000**

Features:
- Drag & drop image upload
- Animated prediction results
- Modern dark theme UI

### 3. Batch Prediction

```bash
python predict_test.py
```

Generates predictions for all test images → `outputs/test_predictions.csv`

## CSV Format

| Column | Type | Description |
|--------|------|-------------|
| lesion_id | string | Lesion ID (used for grouping) |
| image_id | string | Image filename (no extension) |
| dx | string | Diagnosis (akiec, bcc, bkl, df, mel, nv, vasc) |
| age | float | Patient age |
| sex | string | Gender (male/female) |
| localization | string | Body location |

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTIMODAL MODEL                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌───────────────────────────────────┐ │
│  │  Image Branch   │  │        Metadata Branch            │ │
│  │                 │  │                                   │ │
│  │ EfficientNetV2S │  │ Age: Normalize → MLP(32→64)       │ │
│  │ (384×384×3)     │  │ Sex: Embedding(8)                 │ │
│  │       ↓         │  │ Loc: Embedding(16)                │ │
│  │  1280 features  │  │           ↓                       │ │
│  │                 │  │     Concat → Dense(128)           │ │
│  └────────┬────────┘  └─────────────────┬─────────────────┘ │
│           │                             │                   │
│           └─────────────┬───────────────┘                   │
│                         ↓                                   │
│                   Concatenate (1408)                        │
│                         ↓                                   │
│                Dense(512) → Dense(128) → Dense(7)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 384×384 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Initial LR | 1e-4 |
| Fine-tune LR | 1e-5 |
| Frozen Epochs | 5 |
| Fine-tune Epochs | 30 |
| Early Stopping | patience=5 |

### Two-Phase Training
1. **Phase 1**: Backbone frozen, only head trains (5 epochs)
2. **Phase 2**: Last 30% of backbone unfrozen, fine-tune with low LR (30 epochs)

## Data Leakage Prevention

- Uses **StratifiedGroupKFold** with `lesion_id` as group
- Same lesion's images never split between train and validation
- Overlap check performed after split

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No CSV files found" | Check folder paths |
| "Image not found" | Ensure images have `.jpg` or `.png` extension |
| GPU memory error | Reduce `BATCH_SIZE` to 16 or 8 |
| CUDA not available | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

## Notes

- Missing `age` values are filled with training median
- Missing `sex` and `localization` are filled with "unknown"
- Images are normalized with ImageNet mean/std
- Mixed precision (FP16) used for faster training

## Citations & References

### Dataset

**HAM10000 Dataset** - Skin Cancer MNIST  
[Kaggle: Skin Cancer MNIST HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

> Tschandl P, Rosendahl C, Kittler H. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions". *Sci Data*. 2018;5:180161.

### Model Architecture

**EfficientNetV2** - Model implementation reference  
 [GitHub: da2so/efficientnetv2](https://github.com/da2so/efficientnetv2)

> Tan M, Le Q. "EfficientNetV2: Smaller Models and Faster Training". *ICML 2021*.

## License

This project is for educational and research purposes.

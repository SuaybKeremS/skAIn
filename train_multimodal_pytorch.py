#!/usr/bin/env python3
"""
HAM10000 Multimodal Classification Training Script (PyTorch)
Input: Image + Age + Sex + Localization
Output: dx (7 classes: akiec, bcc, bkl, df, mel, nv, vasc)
"""

import os
import json
import glob
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# For EfficientNetV2
import timm

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Suppress warnings
warnings.filterwarnings('ignore')

# ===================== GPU CONFIGURATION =====================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print("⚠ No GPU detected. Training will run on CPU.")

print(f"✓ Using device: {DEVICE}")

# ===================== SEED & DETERMINISM =====================
SEED = 42

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

# ===================== CONFIGURATION =====================
TRAIN_IMAGE_DIR = "dataset_train/image/"
TRAIN_TEXT_DIR = "dataset_train/text/"
TEST_IMAGE_DIR = "dataset_test/image/"
TEST_TEXT_DIR = "dataset_test/text/"
OUTPUT_DIR = "outputs/"

IMG_SIZE = 384
BATCH_SIZE = 32  # Adjust based on GPU memory
INITIAL_EPOCHS = 5  # Frozen backbone epochs
FINETUNE_EPOCHS = 20  # Fine-tune epochs
INITIAL_LR = 1e-4
FINETUNE_LR = 1e-5
NUM_WORKERS = 4  # DataLoader workers

# Class names in fixed order
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)

# ===================== UTILITY FUNCTIONS =====================

def find_largest_csv(directory):
    """Find the CSV file with the most rows in a directory."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    max_rows = -1
    selected_csv = None
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if len(df) > max_rows:
                max_rows = len(df)
                selected_csv = csv_path
        except Exception as e:
            print(f"Warning: Could not read {csv_path}: {e}")
    
    if selected_csv is None:
        raise FileNotFoundError(f"No valid CSV files found in {directory}")
    
    print(f"Selected CSV: {selected_csv} ({max_rows} rows)")
    return selected_csv


def find_image_path(image_id, image_dir):
    """Find image file path by image_id (handles .jpg and .png)."""
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
        path = os.path.join(image_dir, f"{image_id}.{ext}")
        if os.path.exists(path):
            return path
    return None


def load_and_preprocess_data(csv_path, image_dir, is_train=True):
    """Load CSV and preprocess data."""
    df = pd.read_csv(csv_path, dtype={
        'lesion_id': str,
        'image_id': str,
        'dx': str,
        'dx_type': str,
        'age': float,
        'sex': str,
        'localization': str
    })
    
    # Fill missing sex/localization with "unknown"
    df['sex'] = df['sex'].fillna('unknown').replace('', 'unknown')
    df['localization'] = df['localization'].fillna('unknown').replace('', 'unknown')
    
    # Verify image paths exist
    valid_indices = []
    image_paths = []
    
    for idx, row in df.iterrows():
        img_path = find_image_path(row['image_id'], image_dir)
        if img_path:
            valid_indices.append(idx)
            image_paths.append(img_path)
        else:
            print(f"Warning: Image not found for image_id={row['image_id']}, skipping...")
    
    df = df.loc[valid_indices].reset_index(drop=True)
    df['image_path'] = image_paths
    
    if is_train:
        if 'dx' not in df.columns or df['dx'].isna().all():
            raise ValueError("Training data must have 'dx' column with values")
        df = df.dropna(subset=['dx'])
    
    print(f"Loaded {len(df)} valid samples from {csv_path}")
    return df


def create_label_mapping(train_df):
    """Create dx label mapping (alphabetical order)."""
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def compute_age_median(train_df):
    """Compute median age from training data."""
    valid_ages = train_df['age'].dropna()
    if len(valid_ages) > 0:
        return valid_ages.median()
    return 50.0


def fill_missing_age(df, median_age):
    """Fill missing age values with median."""
    df['age'] = df['age'].fillna(median_age)
    return df


def get_vocabularies(train_df):
    """Extract vocabularies for sex and localization from training data."""
    sex_vocab = train_df['sex'].unique().tolist()
    loc_vocab = train_df['localization'].unique().tolist()
    
    if 'unknown' not in sex_vocab:
        sex_vocab.append('unknown')
    if 'unknown' not in loc_vocab:
        loc_vocab.append('unknown')
    
    return sex_vocab, loc_vocab


# ===================== DATA SPLIT (NO LEAKAGE) =====================

def split_data_no_leakage(df, label_to_idx, n_splits=5, val_split=0):
    """Split data ensuring no lesion_id leakage between train and validation."""
    df['label'] = df['dx'].map(label_to_idx)
    groups = df['lesion_id'].values
    labels = df['label'].values
    
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = list(sgkf.split(df, labels, groups))
        train_idx, val_idx = splits[val_split]
        print("Using StratifiedGroupKFold for split")
    except Exception as e:
        print(f"StratifiedGroupKFold failed ({e}), using GroupShuffleSplit")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, val_idx = next(gss.split(df, labels, groups))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Verify no leakage
    train_lesions = set(train_df['lesion_id'].unique())
    val_lesions = set(val_df['lesion_id'].unique())
    overlap = train_lesions.intersection(val_lesions)
    
    if overlap:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(overlap)} lesion_ids in both train and val")
    
    print(f"Train: {len(train_df)} samples, {len(train_lesions)} unique lesions")
    print(f"Val: {len(val_df)} samples, {len(val_lesions)} unique lesions")
    print("No lesion_id leakage detected ✓")
    
    return train_df, val_df


# ===================== PYTORCH DATASET =====================

class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal skin lesion classification."""
    
    def __init__(self, df, label_to_idx, sex_to_idx, loc_to_idx, age_mean, age_std, 
                 img_size=384, augment=False):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.sex_to_idx = sex_to_idx
        self.loc_to_idx = loc_to_idx
        self.age_mean = age_mean
        self.age_std = age_std
        self.img_size = img_size
        self.augment = augment
        
        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and transform image
        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)
        
        # Normalize age
        age = (row['age'] - self.age_mean) / (self.age_std + 1e-8)
        age = torch.tensor([age], dtype=torch.float32)
        
        # Encode sex and localization
        sex = self.sex_to_idx.get(row['sex'], self.sex_to_idx.get('unknown', 0))
        sex = torch.tensor(sex, dtype=torch.long)
        
        loc = self.loc_to_idx.get(row['localization'], self.loc_to_idx.get('unknown', 0))
        loc = torch.tensor(loc, dtype=torch.long)
        
        # Label
        label = row['label']
        label = torch.tensor(label, dtype=torch.long)
        
        return img, age, sex, loc, label


# ===================== MODEL ARCHITECTURE =====================

class MultimodalClassifier(nn.Module):
    """Multimodal classifier combining image features with metadata."""
    
    def __init__(self, num_classes, num_sex, num_loc, pretrained=True):
        super().__init__()
        
        # Image backbone: EfficientNetV2-S
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, num_classes=0)
        img_features = self.backbone.num_features  # 1280 for EfficientNetV2-S
        
        # Age branch
        self.age_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Sex embedding
        self.sex_embedding = nn.Embedding(num_sex + 1, 8)  # +1 for unknown
        
        # Localization embedding
        self.loc_embedding = nn.Embedding(num_loc + 1, 16)  # +1 for unknown
        
        # Meta fusion
        meta_features = 64 + 8 + 16  # age + sex + loc
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final fusion
        fusion_features = img_features + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # For freezing/unfreezing backbone
        self.backbone_frozen = False
    
    def freeze_backbone(self):
        """Freeze the image backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
        print("✓ Backbone frozen")
    
    def unfreeze_backbone(self, unfreeze_ratio=0.3):
        """Unfreeze the last portion of the backbone."""
        all_params = list(self.backbone.parameters())
        freeze_until = int(len(all_params) * (1 - unfreeze_ratio))
        
        for i, param in enumerate(all_params):
            if i >= freeze_until:
                param.requires_grad = True
        
        self.backbone_frozen = False
        print(f"✓ Unfroze last {unfreeze_ratio*100:.0f}% of backbone ({len(all_params) - freeze_until} params)")
    
    def forward(self, img, age, sex, loc):
        # Image features
        x_img = self.backbone(img)
        
        # Age features
        x_age = self.age_fc(age)
        
        # Sex features
        x_sex = self.sex_embedding(sex)
        
        # Localization features
        x_loc = self.loc_embedding(loc)
        
        # Combine metadata
        x_meta = torch.cat([x_age, x_sex, x_loc], dim=1)
        x_meta = self.meta_fc(x_meta)
        
        # Final fusion
        x = torch.cat([x_img, x_meta], dim=1)
        x = self.fusion(x)
        
        return x


# ===================== TRAINING FUNCTIONS =====================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for img, age, sex, loc, labels in pbar:
        img = img.to(device)
        age = age.to(device)
        sex = sex.to(device)
        loc = loc.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(img, age, sex, loc)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(img, age, sex, loc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img, age, sex, loc, labels in tqdm(dataloader, desc="Validating"):
            img = img.to(device)
            age = age.to(device)
            sex = sex.to(device)
            loc = loc.to(device)
            labels = labels.to(device)
            
            outputs = model(img, age, sex, loc)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


# ===================== MAIN TRAINING FUNCTION =====================

def main():
    print("=" * 60)
    print("HAM10000 Multimodal Classification Training (PyTorch)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ---- Step 1: Load Training Data ----
    print("\n[1/8] Loading training data...")
    train_csv = find_largest_csv(TRAIN_TEXT_DIR)
    train_df = load_and_preprocess_data(train_csv, TRAIN_IMAGE_DIR, is_train=True)
    
    # ---- Step 2: Create Label Mapping ----
    print("\n[2/8] Creating label mapping...")
    label_to_idx, idx_to_label = create_label_mapping(train_df)
    
    # Save label mapping
    label_map = {
        'label_to_idx': label_to_idx,
        'idx_to_label': {str(k): v for k, v in idx_to_label.items()}
    }
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Label mapping saved: {label_to_idx}")
    
    # ---- Step 3: Compute Age Stats and Fill Missing ----
    print("\n[3/8] Processing age values...")
    age_median = compute_age_median(train_df)
    print(f"Age median: {age_median}")
    train_df = fill_missing_age(train_df, age_median)
    age_mean = train_df['age'].mean()
    age_std = train_df['age'].std()
    
    # Save age stats for prediction
    with open(os.path.join(OUTPUT_DIR, "age_stats.json"), 'w') as f:
        json.dump({'age_median': age_median, 'age_mean': age_mean, 'age_std': age_std}, f)
    
    # ---- Step 4: Get Vocabularies ----
    print("\n[4/8] Extracting vocabularies...")
    sex_vocab, loc_vocab = get_vocabularies(train_df)
    sex_to_idx = {s: i for i, s in enumerate(sex_vocab)}
    loc_to_idx = {l: i for i, l in enumerate(loc_vocab)}
    print(f"Sex vocab ({len(sex_vocab)}): {sex_vocab}")
    print(f"Localization vocab ({len(loc_vocab)}): {loc_vocab}")
    
    # Save vocabularies
    with open(os.path.join(OUTPUT_DIR, "vocabularies.json"), 'w') as f:
        json.dump({
            'sex_vocab': sex_vocab, 
            'loc_vocab': loc_vocab,
            'sex_to_idx': sex_to_idx,
            'loc_to_idx': loc_to_idx
        }, f, indent=2)
    
    # ---- Step 5: Split Data (No Leakage) ----
    print("\n[5/8] Splitting data (no lesion_id leakage)...")
    train_split_df, val_df = split_data_no_leakage(train_df, label_to_idx)
    
    # ---- Step 6: Compute Class Weights ----
    print("\n[6/8] Computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_split_df['label'].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # ---- Step 7: Create Datasets and DataLoaders ----
    print("\n[7/8] Creating datasets and dataloaders...")
    
    train_dataset = MultimodalDataset(
        train_split_df, label_to_idx, sex_to_idx, loc_to_idx,
        age_mean, age_std, IMG_SIZE, augment=True
    )
    val_dataset = MultimodalDataset(
        val_df, label_to_idx, sex_to_idx, loc_to_idx,
        age_mean, age_std, IMG_SIZE, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ---- Step 8: Build Model ----
    print("\n[8/8] Building model...")
    model = MultimodalClassifier(
        num_classes=NUM_CLASSES,
        num_sex=len(sex_vocab),
        num_loc=len(loc_vocab),
        pretrained=True
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training history
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    best_f1 = 0.0
    
    # ---- Training Phase 1: Frozen Backbone ----
    print("\n" + "=" * 60)
    print("PHASE 1: Training with frozen backbone")
    print("=" * 60)
    
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=INITIAL_LR)
    
    for epoch in range(INITIAL_EPOCHS):
        print(f"\nEpoch {epoch+1}/{INITIAL_EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        
        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"✓ New best model saved (F1: {val_f1:.4f})")
    
    # ---- Training Phase 2: Fine-tune ----
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning backbone")
    print("=" * 60)
    
    model.unfreeze_backbone(unfreeze_ratio=0.3)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR)
    
    patience = 5
    patience_counter = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        print(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        
        # Save history
        history['epoch'].append(INITIAL_EPOCHS + epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"✓ New best model saved (F1: {val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # ---- Save Training History ----
    print("\nSaving training history...")
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)
    
    # ---- Final Evaluation ----
    print("\n" + "=" * 60)
    print("Final Validation Evaluation")
    print("=" * 60)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    val_loss, val_acc, val_f1, y_pred, y_true = validate(model, val_loader, criterion, DEVICE)
    
    # Metrics
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nMacro F1 Score: {val_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "val_metrics.txt"), 'w') as f:
        f.write(f"Macro F1 Score: {val_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best model saved to: {os.path.join(OUTPUT_DIR, 'best_model.pt')}")
    print(f"Best Macro F1: {best_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

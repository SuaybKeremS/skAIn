#!/usr/bin/env python3
"""
HAM10000 Multimodal Classification Prediction Script (PyTorch)
Loads the best trained model and generates predictions for test data.
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# ===================== GPU CONFIGURATION =====================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ No GPU detected - running on CPU")

print(f"✓ Using device: {DEVICE}")

# ===================== CONFIGURATION =====================
TEST_IMAGE_DIR = "dataset_test/image/"
TEST_TEXT_DIR = "dataset_test/text/"
OUTPUT_DIR = "outputs/"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4

# Class names in fixed order (must match training)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)


# ===================== MODEL ARCHITECTURE =====================
# Must match training architecture exactly
import timm
import torch.nn as nn

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, num_sex, num_loc, pretrained=False):
        super().__init__()
        
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=pretrained, num_classes=0)
        img_features = self.backbone.num_features
        
        self.age_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        self.sex_embedding = nn.Embedding(num_sex + 1, 8)
        self.loc_embedding = nn.Embedding(num_loc + 1, 16)
        
        meta_features = 64 + 8 + 16
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        fusion_features = img_features + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, img, age, sex, loc):
        x_img = self.backbone(img)
        x_age = self.age_fc(age)
        x_sex = self.sex_embedding(sex)
        x_loc = self.loc_embedding(loc)
        x_meta = torch.cat([x_age, x_sex, x_loc], dim=1)
        x_meta = self.meta_fc(x_meta)
        x = torch.cat([x_img, x_meta], dim=1)
        x = self.fusion(x)
        return x


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


def load_test_data(csv_path, image_dir, age_median):
    """Load test CSV and preprocess data."""
    df = pd.read_csv(csv_path, dtype={
        'lesion_id': str,
        'image_id': str,
        'dx': str,
        'dx_type': str,
        'age': float,
        'sex': str,
        'localization': str
    })
    
    # Fill missing values
    df['sex'] = df['sex'].fillna('unknown').replace('', 'unknown')
    df['localization'] = df['localization'].fillna('unknown').replace('', 'unknown')
    df['age'] = df['age'].fillna(age_median)
    
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
    
    print(f"Loaded {len(df)} valid test samples")
    return df


# ===================== DATASET =====================

class TestDataset(Dataset):
    def __init__(self, df, sex_to_idx, loc_to_idx, age_mean, age_std, img_size=384):
        self.df = df.reset_index(drop=True)
        self.sex_to_idx = sex_to_idx
        self.loc_to_idx = loc_to_idx
        self.age_mean = age_mean
        self.age_std = age_std
        
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
        
        return img, age, sex, loc, row['image_id']


def load_model_and_configs():
    """Load the trained model and configuration files."""
    # Load vocabularies
    vocab_path = os.path.join(OUTPUT_DIR, "vocabularies.json")
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    sex_to_idx = vocab_data['sex_to_idx']
    loc_to_idx = vocab_data['loc_to_idx']
    
    # Load age stats
    age_stats_path = os.path.join(OUTPUT_DIR, "age_stats.json")
    with open(age_stats_path, 'r') as f:
        age_data = json.load(f)
    age_mean = age_data['age_mean']
    age_std = age_data['age_std']
    age_median = age_data.get('age_median', age_mean)
    
    # Load label mapping
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    idx_to_label = {int(k): v for k, v in label_map['idx_to_label'].items()}
    
    # Build model
    model = MultimodalClassifier(
        num_classes=NUM_CLASSES,
        num_sex=len(sex_to_idx),
        num_loc=len(loc_to_idx),
        pretrained=False
    )
    
    # Load weights
    print(f"Loading model from: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    return model, idx_to_label, sex_to_idx, loc_to_idx, age_mean, age_std, age_median


def predict(model, dataloader):
    """Generate predictions for all test samples."""
    all_predictions = []
    all_probabilities = []
    all_image_ids = []
    
    model.eval()
    with torch.no_grad():
        for img, age, sex, loc, image_ids in tqdm(dataloader, desc="Predicting"):
            img = img.to(DEVICE)
            age = age.to(DEVICE)
            sex = sex.to(DEVICE)
            loc = loc.to(DEVICE)
            
            outputs = model(img, age, sex, loc)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            all_image_ids.extend(image_ids)
    
    return all_image_ids, all_predictions, all_probabilities


def main():
    print("=" * 60)
    print("HAM10000 Multimodal Classification - Test Prediction (PyTorch)")
    print("=" * 60)
    
    # ---- Step 1: Load Model and Configs ----
    print("\n[1/4] Loading model and configurations...")
    model, idx_to_label, sex_to_idx, loc_to_idx, age_mean, age_std, age_median = load_model_and_configs()
    
    # ---- Step 2: Load Test Data ----
    print("\n[2/4] Loading test data...")
    test_csv = find_largest_csv(TEST_TEXT_DIR)
    test_df = load_test_data(test_csv, TEST_IMAGE_DIR, age_median)
    
    if len(test_df) == 0:
        print("Error: No valid test samples found!")
        return
    
    # Create dataset and dataloader
    test_dataset = TestDataset(test_df, sex_to_idx, loc_to_idx, age_mean, age_std, IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # ---- Step 3: Generate Predictions ----
    print("\n[3/4] Generating predictions...")
    image_ids, predictions, probabilities = predict(model, test_loader)
    
    # ---- Step 4: Save Results ----
    print("\n[4/4] Saving predictions...")
    
    # Create results dataframe
    results = pd.DataFrame({
        'image_id': image_ids,
        'predicted_dx': [idx_to_label[p] for p in predictions],
    })
    
    # Add probability columns for each class
    probs_array = np.array(probabilities)
    for i, class_name in enumerate(CLASS_NAMES):
        results[f'p_{class_name}'] = probs_array[:, i]
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    results.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print(f"Results saved to: {output_path}")
    print(f"Total predictions: {len(results)}")
    print("\nPrediction distribution:")
    print(results['predicted_dx'].value_counts())
    print("=" * 60)
    
    # Show sample predictions
    print("\nSample predictions (first 10):")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

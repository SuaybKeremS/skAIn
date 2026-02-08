#!/usr/bin/env python3
"""
HAM10000 Multimodal Classification Prediction Script
Loads the best trained model and generates predictions for test data.
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===================== GPU CONFIGURATION =====================
def configure_gpu():
    """Configure TensorFlow to use GPU with memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU enabled: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU detected - running on CPU")

configure_gpu()

# ===================== CONFIGURATION =====================
TEST_IMAGE_DIR = "dataset_test/image/"
TEST_TEXT_DIR = "dataset_test/text/"
OUTPUT_DIR = "outputs/"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.keras")

IMG_SIZE = 384
BATCH_SIZE = 16

# Class names in fixed order (must match training)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


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
    
    # Fill missing sex/localization with "unknown"
    df['sex'] = df['sex'].fillna('unknown').replace('', 'unknown')
    df['localization'] = df['localization'].fillna('unknown').replace('', 'unknown')
    
    # Fill missing age with train median
    df['age'] = df['age'].fillna(age_median)
    
    # Verify image paths exist
    valid_indices = []
    image_paths = []
    skipped = []
    
    for idx, row in df.iterrows():
        img_path = find_image_path(row['image_id'], image_dir)
        if img_path:
            valid_indices.append(idx)
            image_paths.append(img_path)
        else:
            skipped.append(row['image_id'])
            print(f"Warning: Image not found for image_id={row['image_id']}, skipping...")
    
    df = df.loc[valid_indices].reset_index(drop=True)
    df['image_path'] = image_paths
    
    print(f"Loaded {len(df)} valid test samples")
    if skipped:
        print(f"Skipped {len(skipped)} samples due to missing images")
    
    return df


def load_model_and_configs():
    """Load the trained model and configuration files."""
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    
    # Load label mapping
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    idx_to_label = {int(k): v for k, v in label_map['idx_to_label'].items()}
    print(f"Label mapping loaded: {idx_to_label}")
    
    # Load age median
    age_median_path = os.path.join(OUTPUT_DIR, "age_median.json")
    with open(age_median_path, 'r') as f:
        age_data = json.load(f)
    age_median = age_data['age_median']
    print(f"Age median: {age_median}")
    
    return model, idx_to_label, age_median


def prepare_single_sample(row, img_size):
    """Prepare a single sample for prediction."""
    # Load image
    img = keras.utils.load_img(row['image_path'], target_size=(img_size, img_size))
    img = keras.utils.img_to_array(img)
    
    return {
        'image_input': np.expand_dims(img, axis=0).astype(np.float32),
        'age_input': np.array([[row['age']]], dtype=np.float32),
        'sex_input': np.array([row['sex']], dtype=str),
        'localization_input': np.array([row['localization']], dtype=str),
    }


def predict_batch(model, df, img_size, batch_size):
    """Predict in batches for efficiency."""
    all_predictions = []
    all_probabilities = []
    
    num_samples = len(df)
    num_batches = int(np.ceil(num_samples / batch_size))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Prepare batch
        images = []
        ages = []
        sexes = []
        localizations = []
        
        for _, row in batch_df.iterrows():
            try:
                img = keras.utils.load_img(row['image_path'], target_size=(img_size, img_size))
                img = keras.utils.img_to_array(img)
                images.append(img)
                ages.append(row['age'])
                sexes.append(row['sex'])
                localizations.append(row['localization'])
            except Exception as e:
                print(f"Error loading image {row['image_path']}: {e}")
                # Use zeros as placeholder (will handle below)
                images.append(np.zeros((img_size, img_size, 3), dtype=np.float32))
                ages.append(row['age'])
                sexes.append(row['sex'])
                localizations.append(row['localization'])
        
        batch_input = {
            'image_input': np.array(images, dtype=np.float32),
            'age_input': np.array(ages, dtype=np.float32).reshape(-1, 1),
            'sex_input': np.array(sexes, dtype=str),
            'localization_input': np.array(localizations, dtype=str),
        }
        
        # Predict
        probs = model.predict(batch_input, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        all_predictions.extend(preds.tolist())
        all_probabilities.extend(probs.tolist())
        
        # Progress
        print(f"  Batch {batch_idx + 1}/{num_batches} completed", end='\r')
    
    print()  # New line after progress
    return all_predictions, all_probabilities


def main():
    print("=" * 60)
    print("HAM10000 Multimodal Classification - Test Prediction")
    print("=" * 60)
    
    # ---- Step 1: Load Model and Configs ----
    print("\n[1/4] Loading model and configurations...")
    model, idx_to_label, age_median = load_model_and_configs()
    
    # ---- Step 2: Load Test Data ----
    print("\n[2/4] Loading test data...")
    test_csv = find_largest_csv(TEST_TEXT_DIR)
    test_df = load_test_data(test_csv, TEST_IMAGE_DIR, age_median)
    
    if len(test_df) == 0:
        print("Error: No valid test samples found!")
        return
    
    # ---- Step 3: Generate Predictions ----
    print("\n[3/4] Generating predictions...")
    predictions, probabilities = predict_batch(model, test_df, IMG_SIZE, BATCH_SIZE)
    
    # ---- Step 4: Save Results ----
    print("\n[4/4] Saving predictions...")
    
    # Create results dataframe
    results = pd.DataFrame({
        'image_id': test_df['image_id'].values,
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

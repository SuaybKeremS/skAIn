#!/usr/bin/env python3
"""
HAM10000 Multimodal Classification Training Script
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
<<<<<<< HEAD
import tensorflow as tf
=======

# ===================== GPU CONFIGURATION =====================
# Must be set BEFORE importing tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only GPU 1 is visible
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

# Log device placement to verify GPU usage
tf.debugging.set_log_device_placement(False)  # Set to True to debug

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU 0 (the only visible one after CUDA_VISIBLE_DEVICES='1')
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        # Allocate fixed 7GB memory to force GPU usage
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]  # 7GB of 8GB
        )
        print(f"✓ GPU configured: {gpus[0].name}")
        print(f"✓ Memory allocated: 7GB (fixed)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU detected! Check CUDA installation.")
    print("  Run: nvidia-smi to verify GPU is available")

# Enable Mixed Precision for faster GPU training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✓ Mixed Precision (float16) enabled")
>>>>>>> 6dabdb4 (Change of Tensorflow to PyTorch)
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

<<<<<<< HEAD
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

=======
>>>>>>> 6dabdb4 (Change of Tensorflow to PyTorch)
# ===================== SEED & DETERMINISM =====================
SEED = 42

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Enable deterministic operations if possible
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass

set_seeds()

# ===================== CONFIGURATION =====================
TRAIN_IMAGE_DIR = "dataset_train/image/"
TRAIN_TEXT_DIR = "dataset_train/text/"
TEST_IMAGE_DIR = "dataset_test/image/"
TEST_TEXT_DIR = "dataset_test/text/"
OUTPUT_DIR = "outputs/"

IMG_SIZE = 384
<<<<<<< HEAD
BATCH_SIZE = 16  # Adjust based on GPU memory
=======
BATCH_SIZE = 32  # Increased for better GPU utilization (8GB VRAM)
>>>>>>> 6dabdb4 (Change of Tensorflow to PyTorch)
INITIAL_EPOCHS = 5  # Frozen backbone epochs
FINETUNE_EPOCHS = 20  # Fine-tune epochs
INITIAL_LR = 1e-4
FINETUNE_LR = 1e-5

# Class names in fixed order
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
        # dx is required for training
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
    return 50.0  # Default fallback


def fill_missing_age(df, median_age):
    """Fill missing age values with median."""
    df['age'] = df['age'].fillna(median_age)
    return df


def get_vocabularies(train_df):
    """Extract vocabularies for sex and localization from training data."""
    sex_vocab = train_df['sex'].unique().tolist()
    loc_vocab = train_df['localization'].unique().tolist()
    
    # Ensure 'unknown' is in vocab
    if 'unknown' not in sex_vocab:
        sex_vocab.append('unknown')
    if 'unknown' not in loc_vocab:
        loc_vocab.append('unknown')
    
    return sex_vocab, loc_vocab


# ===================== DATA SPLIT (NO LEAKAGE) =====================

def split_data_no_leakage(df, label_to_idx, n_splits=5, val_split=0):
    """
    Split data ensuring no lesion_id leakage between train and validation.
    Uses StratifiedGroupKFold if available.
    """
    df['label'] = df['dx'].map(label_to_idx)
    groups = df['lesion_id'].values
    labels = df['label'].values
    
    try:
        # Try StratifiedGroupKFold
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


# ===================== DATA AUGMENTATION =====================

def create_augmentation_layer():
    """Create data augmentation sequential layer."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),  # ±36 degrees
        layers.RandomZoom(0.1),  # ±10%
        layers.RandomContrast(0.1),  # ±10%
    ], name="augmentation")


# ===================== TF DATA PIPELINE =====================

class MultimodalDataGenerator(keras.utils.Sequence):
    """Custom data generator for multimodal input."""
    
    def __init__(self, df, label_to_idx, batch_size, img_size, augment=False, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.augmentation = create_augmentation_layer() if augment else None
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        images = []
        ages = []
        sexes = []
        localizations = []
        labels = []
        
        for _, row in batch_df.iterrows():
            # Load and preprocess image
            img = keras.utils.load_img(row['image_path'], target_size=(self.img_size, self.img_size))
            img = keras.utils.img_to_array(img)
            images.append(img)
            
            # Metadata
            ages.append(row['age'])
            sexes.append(row['sex'])
            localizations.append(row['localization'])
            
            # Label
            if 'label' in row:
                labels.append(row['label'])
            else:
                labels.append(self.label_to_idx.get(row.get('dx', ''), 0))
        
        images = np.array(images, dtype=np.float32)
        
        # Apply augmentation if enabled
        if self.augment and self.augmentation:
            images = self.augmentation(images, training=True)
        
        # Prepare inputs - use tf.constant for strings to avoid dtype issues
        inputs = {
            'image_input': images,
            'age_input': np.array(ages, dtype=np.float32).reshape(-1, 1),
            'sex_input': tf.constant(sexes, dtype=tf.string),
            'localization_input': tf.constant(localizations, dtype=tf.string),
        }
        
        labels = np.array(labels, dtype=np.int32)
        
        return inputs, labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_dataset(df, label_to_idx, batch_size, img_size, augment=False, shuffle=True):
    """Create tf.data.Dataset from dataframe."""
    
    def generator():
        indices = np.arange(len(df))
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            row = df.iloc[idx]
            
            # Load image
            try:
                img = tf.io.read_file(row['image_path'])
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, [img_size, img_size])
                img = tf.cast(img, tf.float32)
            except Exception as e:
                print(f"Error loading image {row['image_path']}: {e}")
                continue
            
            age = np.float32(row['age'])
            sex = str(row['sex'])
            localization = str(row['localization'])
            label = np.int32(row['label'])
            
            yield (img, age, sex, localization), label
    
    output_signature = (
        (
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    def prepare_inputs(inputs, label):
        img, age, sex, loc = inputs
        return {
            'image_input': img,
            'age_input': tf.reshape(age, (1,)),
            'sex_input': sex,
            'localization_input': loc,
        }, label
    
    dataset = dataset.map(prepare_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        aug_layer = create_augmentation_layer()
        def augment_image(inputs, label):
            inputs['image_input'] = aug_layer(tf.expand_dims(inputs['image_input'], 0), training=True)[0]
            return inputs, label
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ===================== MODEL ARCHITECTURE =====================

def build_multimodal_model(sex_vocab, loc_vocab, age_normalizer):
    """Build multimodal classification model."""
    
    # ---- Image Branch ----
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    
    # EfficientNetV2-S backbone (pretrained)
    backbone = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_tensor=image_input,
        pooling=None
    )
    backbone.trainable = False  # Initially frozen
    
    x_img = backbone.output
    x_img = layers.GlobalAveragePooling2D(name='img_gap')(x_img)
    x_img = layers.Dropout(0.3, name='img_dropout')(x_img)
    
    # ---- Age Branch ----
    age_input = layers.Input(shape=(1,), dtype=tf.float32, name='age_input')
    x_age = age_normalizer(age_input)
    x_age = layers.Dense(32, activation='relu', name='age_dense1')(x_age)
    x_age = layers.Dense(64, activation='relu', name='age_dense2')(x_age)
    
    # ---- Sex Branch ----
    sex_input = layers.Input(shape=(), dtype=tf.string, name='sex_input')
    sex_lookup = layers.StringLookup(vocabulary=sex_vocab, oov_token='unknown', name='sex_lookup')
    x_sex = sex_lookup(sex_input)
    x_sex = layers.Embedding(input_dim=len(sex_vocab) + 2, output_dim=8, name='sex_embedding')(x_sex)
    x_sex = layers.Flatten(name='sex_flatten')(x_sex)
    
    # ---- Localization Branch ----
    loc_input = layers.Input(shape=(), dtype=tf.string, name='localization_input')
    loc_lookup = layers.StringLookup(vocabulary=loc_vocab, oov_token='unknown', name='loc_lookup')
    x_loc = loc_lookup(loc_input)
    x_loc = layers.Embedding(input_dim=len(loc_vocab) + 2, output_dim=16, name='loc_embedding')(x_loc)
    x_loc = layers.Flatten(name='loc_flatten')(x_loc)
    
    # ---- Meta Fusion ----
    x_meta = layers.Concatenate(name='meta_concat')([x_age, x_sex, x_loc])
    x_meta = layers.Dense(128, activation='relu', name='meta_dense')(x_meta)
    x_meta = layers.Dropout(0.2, name='meta_dropout')(x_meta)
    
    # ---- Final Fusion ----
    x_fusion = layers.Concatenate(name='fusion_concat')([x_img, x_meta])
    x_fusion = layers.Dense(512, activation='relu', name='fusion_dense1')(x_fusion)
    x_fusion = layers.Dropout(0.3, name='fusion_dropout')(x_fusion)
    x_fusion = layers.Dense(128, activation='relu', name='fusion_dense2')(x_fusion)
    
    # Output
    output = layers.Dense(len(CLASS_NAMES), activation='softmax', name='output')(x_fusion)
    
    model = Model(
        inputs=[image_input, age_input, sex_input, loc_input],
        outputs=output,
        name='multimodal_classifier'
    )
    
    return model, backbone


# ===================== CUSTOM CALLBACKS =====================

class MacroF1Callback(Callback):
    """Callback to compute macro F1 score on validation set."""
    
    def __init__(self, val_data, val_labels):
        super().__init__()
        self.val_data = val_data
        self.val_labels = val_labels
        self.best_f1 = 0.0
        self.history = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred_probs = self.model.predict(self.val_data, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Compute macro F1
        macro_f1 = f1_score(self.val_labels, y_pred, average='macro')
        self.history.append(macro_f1)
        
        if logs is not None:
            logs['val_macro_f1'] = macro_f1
        
        print(f" - val_macro_f1: {macro_f1:.4f}")
        
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1


class F1ModelCheckpoint(Callback):
    """Custom checkpoint that saves based on macro F1."""
    
    def __init__(self, filepath, val_data, val_labels):
        super().__init__()
        self.filepath = filepath
        self.val_data = val_data
        self.val_labels = val_labels
        self.best_f1 = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        y_pred_probs = self.model.predict(self.val_data, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        macro_f1 = f1_score(self.val_labels, y_pred, average='macro')
        
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            print(f"\nEpoch {epoch+1}: val_macro_f1 improved to {macro_f1:.4f}, saving model...")
            self.model.save(self.filepath)


# ===================== PREPARE VALIDATION DATA =====================

def prepare_validation_data(val_df, img_size):
    """Prepare validation data as numpy arrays for callbacks."""
    images = []
    ages = []
    sexes = []
    localizations = []
    labels = []
    
    for _, row in val_df.iterrows():
        try:
            img = keras.utils.load_img(row['image_path'], target_size=(img_size, img_size))
            img = keras.utils.img_to_array(img)
            images.append(img)
            ages.append(row['age'])
            sexes.append(row['sex'])
            localizations.append(row['localization'])
            labels.append(row['label'])
        except Exception as e:
            print(f"Warning: Could not load {row['image_path']}: {e}")
    
    # Use tf.constant for strings to avoid dtype issues
    val_data = {
        'image_input': np.array(images, dtype=np.float32),
        'age_input': np.array(ages, dtype=np.float32).reshape(-1, 1),
        'sex_input': tf.constant(sexes, dtype=tf.string),
        'localization_input': tf.constant(localizations, dtype=tf.string),
    }
    val_labels = np.array(labels, dtype=np.int32)
    
    return val_data, val_labels


# ===================== MAIN TRAINING FUNCTION =====================

def main():
    print("=" * 60)
    print("HAM10000 Multimodal Classification Training")
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
    
    # ---- Step 3: Compute Age Median and Fill Missing ----
    print("\n[3/8] Processing age values...")
    age_median = compute_age_median(train_df)
    print(f"Age median: {age_median}")
    train_df = fill_missing_age(train_df, age_median)
    
    # Save age median for prediction
    with open(os.path.join(OUTPUT_DIR, "age_median.json"), 'w') as f:
        json.dump({'age_median': age_median}, f)
    
    # ---- Step 4: Get Vocabularies ----
    print("\n[4/8] Extracting vocabularies...")
    sex_vocab, loc_vocab = get_vocabularies(train_df)
    print(f"Sex vocab ({len(sex_vocab)}): {sex_vocab}")
    print(f"Localization vocab ({len(loc_vocab)}): {loc_vocab}")
    
    # Save vocabularies
    with open(os.path.join(OUTPUT_DIR, "vocabularies.json"), 'w') as f:
        json.dump({'sex_vocab': sex_vocab, 'loc_vocab': loc_vocab}, f, indent=2)
    
    # ---- Step 5: Split Data (No Leakage) ----
    print("\n[5/8] Splitting data (no lesion_id leakage)...")
    train_split_df, val_df = split_data_no_leakage(train_df, label_to_idx)
    
    # ---- Step 6: Compute Class Weights ----
    print("\n[6/8] Computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(CLASS_NAMES)),
        y=train_split_df['label'].values
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # ---- Step 7: Build Model ----
    print("\n[7/8] Building model...")
    
    # Create age normalizer and adapt
    age_normalizer = layers.Normalization(axis=-1)
    age_normalizer.adapt(train_split_df['age'].values.reshape(-1, 1))
    
    model, backbone = build_multimodal_model(sex_vocab, loc_vocab, age_normalizer)
    model.summary()
    
<<<<<<< HEAD
    # ---- Step 8: Prepare Data Generators ----
    print("\n[8/8] Preparing data generators...")
    
    train_gen = MultimodalDataGenerator(
        train_split_df, label_to_idx, BATCH_SIZE, IMG_SIZE,
        augment=True, shuffle=True
    )
    val_gen = MultimodalDataGenerator(
=======
    # ---- Step 8: Prepare Data Pipelines ----
    print("\n[8/8] Preparing tf.data pipelines (GPU-optimized)...")
    
    # Use tf.data pipeline instead of Keras Sequence for better GPU utilization
    train_dataset = create_tf_dataset(
        train_split_df, label_to_idx, BATCH_SIZE, IMG_SIZE,
        augment=True, shuffle=True
    )
    val_dataset = create_tf_dataset(
>>>>>>> 6dabdb4 (Change of Tensorflow to PyTorch)
        val_df, label_to_idx, BATCH_SIZE, IMG_SIZE,
        augment=False, shuffle=False
    )
    
    # Prepare validation data for callbacks
    val_data, val_labels = prepare_validation_data(val_df, IMG_SIZE)
    
    # ---- Training Phase 1: Frozen Backbone ----
    print("\n" + "=" * 60)
    print("PHASE 1: Training with frozen backbone")
    print("=" * 60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    f1_callback = MacroF1Callback(val_data, val_labels)
    checkpoint_callback = F1ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.keras"),
        val_data, val_labels
    )
    
    history1 = model.fit(
<<<<<<< HEAD
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
=======
        train_dataset,
        epochs=INITIAL_EPOCHS,
        validation_data=val_dataset,
>>>>>>> 6dabdb4 (Change of Tensorflow to PyTorch)
        class_weight=class_weight_dict,
        callbacks=[f1_callback, checkpoint_callback],
        verbose=1
    )
    
    # ---- Training Phase 2: Fine-tune ----
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning backbone")
    print("=" * 60)
    
    # Unfreeze the last portion of the backbone
    backbone.trainable = True
    
    # Freeze early layers (keep first 70% frozen)
    num_layers = len(backbone.layers)
    freeze_until = int(num_layers * 0.7)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"Unfroze {num_layers - freeze_until}/{num_layers} backbone layers")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    f1_callback2 = MacroF1Callback(val_data, val_labels)
    checkpoint_callback2 = F1ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.keras"),
        val_data, val_labels
    )
    
    history2 = model.fit(
<<<<<<< HEAD
        train_gen,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_gen,
=======
        train_dataset,
        epochs=FINETUNE_EPOCHS,
        validation_data=val_dataset,
>>>>>>> 6dabdb4 (Change of Tensorflow to PyTorch)
        class_weight=class_weight_dict,
        callbacks=[f1_callback2, checkpoint_callback2, early_stopping],
        verbose=1
    )
    
    # ---- Save Training History ----
    print("\nSaving training history...")
    
    # Combine histories
    full_history = {
        'epoch': list(range(1, len(history1.history['loss']) + len(history2.history['loss']) + 1)),
        'loss': history1.history['loss'] + history2.history['loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'val_macro_f1': f1_callback.history + f1_callback2.history,
    }
    
    history_df = pd.DataFrame(full_history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "train_history.csv"), index=False)
    
    # ---- Final Evaluation ----
    print("\n" + "=" * 60)
    print("Final Validation Evaluation")
    print("=" * 60)
    
    # Load best model
    best_model = keras.models.load_model(os.path.join(OUTPUT_DIR, "best_model.keras"))
    
    # Predictions
    y_pred_probs = best_model.predict(val_data, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    macro_f1 = f1_score(val_labels, y_pred, average='macro')
    report = classification_report(val_labels, y_pred, target_names=CLASS_NAMES)
    cm = confusion_matrix(val_labels, y_pred)
    
    print(f"\nMacro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "val_metrics.txt"), 'w') as f:
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best model saved to: {os.path.join(OUTPUT_DIR, 'best_model.keras')}")
    print(f"Label map saved to: {os.path.join(OUTPUT_DIR, 'label_map.json')}")
    print(f"Training history saved to: {os.path.join(OUTPUT_DIR, 'train_history.csv')}")
    print(f"Validation metrics saved to: {os.path.join(OUTPUT_DIR, 'val_metrics.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

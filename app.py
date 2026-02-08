#!/usr/bin/env python3
"""
Skin Disease Prediction Web Application
Flask backend for serving predictions from the trained PyTorch model.
"""

import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import timm
import io
import base64

app = Flask(__name__)
CORS(app)

# ===================== CONFIGURATION =====================
OUTPUT_DIR = "outputs/"
IMG_SIZE = 384
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Class names
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus (Mole)',
    'vasc': 'Vascular Lesion'
}

# ===================== MODEL ARCHITECTURE =====================
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
        
        self.backbone_frozen = False
    
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

# ===================== LOAD MODEL =====================
model = None
sex_to_idx = {}
loc_to_idx = {}
age_mean = 50.0
age_std = 15.0

def load_model():
    global model, sex_to_idx, loc_to_idx, age_mean, age_std
    
    # Load vocabularies
    vocab_path = os.path.join(OUTPUT_DIR, "vocabularies.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            sex_to_idx = vocab_data.get('sex_to_idx', {'male': 0, 'female': 1, 'unknown': 2})
            loc_to_idx = vocab_data.get('loc_to_idx', {'unknown': 0})
    else:
        sex_to_idx = {'male': 0, 'female': 1, 'unknown': 2}
        loc_to_idx = {'back': 0, 'lower extremity': 1, 'trunk': 2, 'upper extremity': 3, 
                      'abdomen': 4, 'face': 5, 'chest': 6, 'foot': 7, 'unknown': 8,
                      'neck': 9, 'scalp': 10, 'hand': 11, 'ear': 12, 'genital': 13, 'acral': 14}
    
    # Load age stats
    age_stats_path = os.path.join(OUTPUT_DIR, "age_stats.json")
    if os.path.exists(age_stats_path):
        with open(age_stats_path, 'r') as f:
            age_data = json.load(f)
            age_mean = age_data.get('age_mean', 50.0)
            age_std = age_data.get('age_std', 15.0)
    
    # Initialize model
    model = MultimodalClassifier(
        num_classes=len(CLASS_NAMES),
        num_sex=len(sex_to_idx),
        num_loc=len(loc_to_idx),
        pretrained=False
    )
    
    # Load weights if available
    model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✓ Model loaded from {model_path}")
    else:
        print("⚠ No trained model found. Using random weights.")
    
    model = model.to(DEVICE)
    model.eval()
    print(f"✓ Model ready on {DEVICE}")

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== ROUTES =====================
@app.route('/')
def index():
    return render_template('index.html', localizations=list(loc_to_idx.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Get metadata
        age = float(request.form.get('age', 50))
        sex = request.form.get('sex', 'unknown').lower()
        localization = request.form.get('localization', 'unknown').lower()
        
        # Normalize age
        age_normalized = (age - age_mean) / (age_std + 1e-8)
        age_tensor = torch.tensor([[age_normalized]], dtype=torch.float32).to(DEVICE)
        
        # Encode sex and localization
        sex_idx = sex_to_idx.get(sex, sex_to_idx.get('unknown', 0))
        loc_idx = loc_to_idx.get(localization, loc_to_idx.get('unknown', 0))
        
        sex_tensor = torch.tensor([sex_idx], dtype=torch.long).to(DEVICE)
        loc_tensor = torch.tensor([loc_idx], dtype=torch.long).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor, age_tensor, sex_tensor, loc_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Format results
        results = []
        for i, class_name in enumerate(CLASS_NAMES):
            results.append({
                'class': class_name,
                'description': CLASS_DESCRIPTIONS[class_name],
                'probability': float(probabilities[i]) * 100
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': str(DEVICE)})

# ===================== MAIN =====================
if __name__ == '__main__':
    load_model()
    print("\n" + "=" * 50)
    print("🏥 Skin Disease Prediction App")
    print("=" * 50)
    print("Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

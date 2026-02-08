# HAM10000 Multimodal Skin Lesion Classification

Bu proje, HAM10000 veri seti üzerinde **multimodal sınıflandırma** yapmaktadır:
- **Giriş**: Görüntü + yaş (age) + cinsiyet (sex) + lokalizasyon (localization)
- **Çıkış**: dx (7 sınıf: akiec, bcc, bkl, df, mel, nv, vasc)

## 📁 Dosya Yapısı

```
project/
├── dataset_train/
│   ├── image/          # Eğitim görüntüleri (.jpg veya .png)
│   └── text/           # Eğitim CSV dosyası
├── dataset_test/
│   ├── image/          # Test görüntüleri (.jpg veya .png)
│   └── text/           # Test CSV dosyası
├── outputs/            # Çıktılar (otomatik oluşturulur)
│   ├── best_model/     # Kaydedilen en iyi model (SavedModel)
│   ├── label_map.json  # Etiket eşlemesi
│   ├── train_history.csv
│   ├── val_metrics.txt
│   └── test_predictions.csv
├── train_multimodal.py # Eğitim scripti
├── predict_test.py     # Tahmin scripti
├── requirements.txt    # Bağımlılıklar
└── README.md           # Bu dosya
```

## 🔧 Kurulum

```bash
# 1. Sanal ortam oluştur (opsiyonel ama önerilir)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows

# 2. Bağımlılıkları yükle
pip install -r requirements.txt
```

## 🚀 Kullanım

### 1. Eğitim

```bash
python train_multimodal.py
```

Bu komut:
- `dataset_train/text/` içindeki en büyük CSV dosyasını bulur
- Verileri lesion_id bazlı böler (data leakage yok)
- EfficientNetV2-S backbone ile modeli eğitir
- En iyi modeli `outputs/best_model/` dizinine kaydeder
- Eğitim geçmişini ve validasyon metriklerini kaydeder

### 2. Test Tahmini

```bash
python predict_test.py
```

Bu komut:
- Eğitilmiş modeli yükler
- `dataset_test/text/` içindeki CSV'den test verilerini okur
- Her görüntü için tahmin yapar
- Sonuçları `outputs/test_predictions.csv` dosyasına kaydeder

## 📊 CSV Formatı

### Eğitim CSV Kolonları
| Kolon | Tip | Açıklama |
|-------|-----|----------|
| lesion_id | string | Lezyon ID (split için grup) |
| image_id | string | Görüntü dosya adı (uzantısız) |
| dx | string | Tanı (akiec, bcc, bkl, df, mel, nv, vasc) |
| dx_type | string | (Kullanılmıyor) |
| age | float | Yaş |
| sex | string | Cinsiyet (male/female) |
| localization | string | Vücut bölgesi |

### Test CSV Kolonları
Aynı format, ancak `dx` kolonu olmayabilir veya boş olabilir.

## 🏗️ Model Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTIMODAL MODEL                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌───────────────────────────────────┐ │
│  │  Image Branch   │  │        Metadata Branch            │ │
│  │                 │  │                                   │ │
│  │ EfficientNetV2S │  │ Age: Normalization → MLP(32→64)   │ │
│  │ (384x384x3)     │  │ Sex: StringLookup → Embedding(8)  │ │
│  │       ↓         │  │ Loc: StringLookup → Embedding(16) │ │
│  │ GlobalAvgPool2D │  │           ↓                       │ │
│  │   Dropout(0.3)  │  │     Concat → Dense(128)           │ │
│  │                 │  │       Dropout(0.2)                │ │
│  └────────┬────────┘  └─────────────────┬─────────────────┘ │
│           │                             │                   │
│           └─────────────┬───────────────┘                   │
│                         ↓                                   │
│                   Concatenate                               │
│                         ↓                                   │
│                Dense(512, ReLU)                             │
│                 Dropout(0.3)                                │
│                Dense(128, ReLU)                             │
│                         ↓                                   │
│               Dense(7, Softmax)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔒 Data Leakage Koruması

- **StratifiedGroupKFold** kullanılır
- `lesion_id` grup olarak kullanılır
- Aynı lezyonun görüntüleri asla train ve validation'a birlikte gitmez
- Split sonrası overlap kontrolü yapılır

## ⚙️ Eğitim Detayları

| Parametre | Değer |
|-----------|-------|
| Image Size | 384x384 |
| Batch Size | 16 |
| Optimizer | Adam |
| Initial LR | 1e-4 |
| Fine-tune LR | 1e-5 |
| Frozen Epochs | 5 |
| Fine-tune Epochs | 20 |
| Early Stopping | patience=5 |

### İki Aşamalı Eğitim
1. **Aşama 1**: Backbone dondurulmuş, sadece head eğitilir (5 epoch)
2. **Aşama 2**: Backbone'un son %30'u açılır, düşük LR ile fine-tune (20 epoch)

## 📈 Çıktılar

### `outputs/test_predictions.csv`
```csv
image_id,predicted_dx,p_akiec,p_bcc,p_bkl,p_df,p_mel,p_nv,p_vasc
ISIC_0024306,nv,0.01,0.02,0.05,0.01,0.03,0.85,0.03
ISIC_0024307,mel,0.02,0.03,0.10,0.02,0.75,0.05,0.03
...
```

### `outputs/val_metrics.txt`
- Macro F1 Score
- Classification Report
- Confusion Matrix

## 🐛 Hata Giderme

1. **"No CSV files found"**: Klasör yollarını kontrol edin
2. **"Image not found"**: Görüntü dosyalarının `.jpg` veya `.png` uzantılı olduğundan emin olun
3. **GPU bellek hatası**: `BATCH_SIZE` değerini azaltın (8 veya 4)

## 📝 Notlar

- Eksik `age` değerleri train medyanı ile doldurulur
- Eksik `sex` ve `localization` değerleri "unknown" ile doldurulur
- Her image_id birden fazla satırda olabilir (aynı görüntü farklı metadata ile)
- `dx_type` kolonu modelde kullanılmaz

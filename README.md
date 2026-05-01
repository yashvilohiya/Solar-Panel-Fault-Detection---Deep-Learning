
# 🔆 Solar Panel Fault Detection

A deep learning classifier that detects faults in solar panels using transfer learning with **EfficientNetB0**. The model classifies solar panel images into 6 categories and achieves **82.76% test accuracy**.

---

## 📋 Classes

| Class | Description |
|-------|-------------|
| `Bird-drop` | Panels soiled by bird droppings |
| `Clean` | Panels in good working condition |
| `Dusty` | Panels covered with dust or dirt |
| `Electrical-damage` | Panels with electrical faults |
| `Physical-Damage` | Panels with physical cracks or damage |
| `Snow-Covered` | Panels obscured by snow |

---

## 📊 Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **82.76%** |
| Macro F1-Score | 0.8312 |
| Weighted F1-Score | 0.8039 |
| Test Loss | 0.4903 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Bird-drop | 0.79 | 0.73 | 0.76 |
| Clean | 0.75 | 0.71 | 0.73 |
| Dusty | 0.67 | 0.79 | 0.73 |
| Electrical-damage | 0.86 | 0.90 | 0.88 |
| Physical-Damage | 0.91 | 0.77 | 0.83 |
| Snow-Covered | 0.96 | 0.96 | 0.96 |

---

## 🏗️ Model Architecture

- **Backbone**: EfficientNetB0 (pre-trained on ImageNet, frozen)
- **Head**:
  - `GlobalAveragePooling2D`
  - `Dense(256, relu)` → `BatchNorm` → `Dropout(0.5)`
  - `Dense(128, relu)` → `BatchNorm` → `Dropout(0.3)`
  - `Dense(6, softmax)`
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Categorical Crossentropy
- **Class Balancing**: Inverse-frequency class weights

---

## 📁 Project Structure

```
solar-panel-fault-detection/
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
├── .gitignore
├── results/               # Output visualizations
│   ├── class_distribution.png
│   ├── augmented_samples.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── sample_predictions.png
│   └── metrics_dashboard.png
└── Faulty_solar_panel/    # Dataset directory (not included)
    ├── Bird-drop/
    ├── Clean/
    ├── Dusty/
    ├── Electrical-damage/
    ├── Physical-Damage/
    └── Snow-Covered/
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the [Faulty Solar Panel dataset](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) and place it in the project root:

```
Faulty_solar_panel/
├── Bird-drop/      ← place images here
├── Clean/
├── Dusty/
├── Electrical-damage/
├── Physical-Damage/
└── Snow-Covered/
```

### 4. Train the model

```bash
python train.py
```

Training will:
- Auto-split data 80/20 train/val
- Apply data augmentation
- Save the best model to `best_model.keras`
- Save all result plots to `results/`

---

## 📈 Visualizations

### Class Distribution
![Class Distribution](results/class_distribution.png)

### Sample Augmented Training Images
![Augmented Samples](results/augmented_samples.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Training History
![Training History](results/training_history.png)

### Sample Predictions
![Sample Predictions](results/sample_predictions.png)

### Metrics Dashboard
![Metrics Dashboard](results/metrics_dashboard.png)

---

## 🔧 Training Details

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs |
| LR reduction patience | 5 epochs |
| Validation split | 20% |
| Augmentation | Rotation, shift, shear, zoom, flip |

---

## 📦 Dataset Stats

| Class | Total | Train | Val |
|-------|-------|-------|-----|
| Bird-drop | ~206 | ~165 | ~41 |
| Clean | ~190 | ~152 | ~38 |
| Dusty | ~190 | ~152 | ~38 |
| Electrical-damage | ~100 | ~80 | ~20 |
| Physical-Damage | ~65 | ~52 | ~13 |
| Snow-Covered | ~120 | ~96 | ~24 |

Imbalance ratio ≈ **3.17×** → compensated with class weighting.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

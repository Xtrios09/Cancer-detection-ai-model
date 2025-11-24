# 🏥 Multi-Dataset Cancer Detection Model

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Research Only](https://img.shields.io/badge/Use-Research%20Only-red.svg)](https://github.com)

## ⚠️ CRITICAL WARNING - RESEARCH USE ONLY

**THIS MODEL IS NOT APPROVED FOR CLINICAL USE**

This cancer detection AI model is intended for **research and educational purposes ONLY**. It has NOT been:

- ✗ Validated in clinical settings
- ✗ Approved by FDA, EMA, or other regulatory bodies  
- ✗ Tested for safety in medical decision-making
- ✗ Peer-reviewed or published in medical journals
- ✗ Certified for diagnostic accuracy in real-world scenarios

### **DO NOT USE for:**
- ❌ Clinical diagnosis or screening
- ❌ Patient care decisions
- ❌ Medical treatment planning
- ❌ Commercial medical applications
- ❌ Any life-critical healthcare decisions

**If you suspect cancer, consult a qualified medical professional immediately.**

---

## 📋 Project Overview

This project implements an advanced custom Convolutional Neural Network (CNN) architecture designed from scratch for binary cancer classification from medical imaging data. The model combines state-of-the-art deep learning techniques to achieve robust feature extraction and classification.

### 🎯 Key Features

- **Advanced CNN Architecture** with modern deep learning components
- **Multi-scale feature extraction** using Inception-style modules
- **Attention mechanisms** (Channel + Spatial Attention - CBAM)
- **Residual connections** with Squeeze-and-Excitation blocks
- **Transfer learning ready** with customizable architecture
- **Comprehensive data augmentation** pipeline
- **Detailed performance metrics** and visualization tools

---

## 🏗️ Model Architecture

### Architecture Highlights

```
Input (224x224x3 RGB Images)
    ↓
[Stage 1: Initial Feature Extraction]
    ↓
[Stage 2: Residual Block + SE Attention]
    ↓
[Stage 3: Inception Module + CBAM]
    ↓
[Stage 4: Deep Feature Processing]
    ↓
[Global Average Pooling]
    ↓
[Dense Classification Head with Dropout]
    ↓
Output (Binary: Benign/Malignant)
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Residual Connections** | Skip connections for gradient flow and deeper networks |
| **Squeeze-and-Excitation (SE) Blocks** | Channel-wise feature recalibration |
| **Inception Modules** | Multi-scale feature extraction with parallel convolutions |
| **CBAM Attention** | Channel + Spatial attention for focused learning |
| **Global Average Pooling** | Reduces overfitting compared to fully connected layers |
| **Dropout Regularization** | Prevents overfitting in classification head |

### Model Statistics

- **Input Shape:** 224×224×3 (RGB images)
- **Output:** Binary classification (2 classes)
- **Total Parameters:** 2,563,793 (9.78 MB)
- **Trainable Parameters:** 2,551,185 (9.73 MB)
- **Non-trainable Parameters:** 12,608 (49.25 KB)
- **Architecture Depth:** 4 main processing stages
- **Activation Functions:** ReLU, Sigmoid (output)
- **Optimization:** Adam optimizer with learning rate scheduling

---

## 📊 Dataset Information

### Training Data Sources

This model is trained on combined public medical imaging datasets:

- **Total Training Samples:** 61,876 images
- **Total Validation Samples:** 10,918 images
- **Total Test Samples:** 7,400 images
- **Image Format:** RGB images (224×224 pixels)
- **Classes:** Binary (Benign vs Malignant)
- **Class Distribution:**
  - Benign: Higher representation
  - Malignant: Lower representation
- **Class Weights Applied:** {0: 1.628, 1: 0.722} to handle imbalance

### Data Augmentation

Applied augmentations to improve model generalization:

```python
- Rotation: ±20 degrees
- Width/Height Shift: ±20%
- Shear Transformation
- Zoom: ±20%
- Horizontal Flip
- Fill Mode: Nearest neighbor
```

### Data Preprocessing

- **Normalization:** Pixel values scaled to [0, 1]
- **Resizing:** All images resized to 224×224
- **Color Mode:** RGB (3 channels)
- **Batch Size:** 32

---

## 📈 Training Results & Performance

### Training Configuration

- **Total Epochs:** 100 (early stopping may apply)
- **Batch Size:** 32
- **Optimizer:** Adam with learning rate decay
- **Initial Learning Rate:** 0.001
- **Loss Function:** Binary Cross-Entropy
- **Hardware:** CPU (Google Colab/Kaggle GPU recommended)
- **Training Time per Epoch:** ~2.5 hours on CPU (~5 seconds per step)

### Current Training Progress

#### Epoch-by-Epoch Results

| Epoch | Train Acc | Val Acc | Train AUC | Val AUC | Train Loss | Val Loss | Time |
|-------|-----------|---------|-----------|---------|------------|----------|------|
| 1 | 79.11% | 43.88% | 0.8545 | 0.5195 | 0.5827 | 1.5807 | 2h 44m |
| 2 | 87.09% | 61.95% | 0.9257 | 0.6797 | 0.3744 | 1.1490 | 2h 39m |
| 3 | 87.80% | ~65%* | 0.9400 | ~0.70* | 0.3183 | ~1.00* | In progress |

*Estimated based on training trend

### Training Visualizations

Below are the comprehensive training metrics visualized across epochs:

#### 1. Complete Training Dashboard
![Training History](results/plots/training_history_complete.png)
*Figure 1: Comprehensive view of all training metrics including Accuracy, Loss, AUC, Precision, Recall, and Learning Rate*

#### 2. Accuracy Progression
![Accuracy Plot](results/plots/accuracy_plot.png)
*Figure 2: Training vs Validation Accuracy showing model learning progression and potential overfitting*

#### 3. Loss Curves
![Loss Plot](results/plots/loss_plot.png)
*Figure 3: Training vs Validation Loss demonstrating convergence behavior*

#### 4. AUC-ROC Performance
![AUC Plot](results/plots/auc_plot.png)
*Figure 4: Area Under Curve metric showing discrimination capability between classes*

### Performance Analysis

**Current Observations:**
- ✅ **Training accuracy improving:** 79% → 87% → 88%
- ⚠️ **Validation accuracy lagging:** 44% → 62% → ~65%
- ⚠️ **Gap indicates overfitting:** Model memorizing training data
- ✅ **AUC improving steadily:** Shows good class discrimination ability
- ⚠️ **High validation loss:** Suggests need for regularization

**Model Behavior:**
- **Overfitting detected:** Training accuracy (87%) significantly higher than validation (62%)
- **Learning rate decay:** Gradually reducing from 0.001 to prevent overshooting
- **Class imbalance handling:** Class weights applied successfully

### Recommended Improvements

Based on current results, consider:

1. **Add more dropout** (increase from 0.3 to 0.5)
2. **Implement early stopping** (patience=10 epochs)
3. **Increase data augmentation** intensity
4. **Add L2 regularization** to dense layers
5. **Reduce model complexity** if overfitting persists
6. **Collect more diverse training data**

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
TensorFlow >= 2.8.0
Keras >= 2.8.0
NumPy >= 1.21.0
Matplotlib >= 3.4.0
scikit-learn >= 0.24.0
Pillow >= 8.0.0
seaborn >= 0.11.0
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/cancer-detection-model.git
cd cancer-detection-model
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Organize your dataset:**
```
dataset/
├── train/
│   ├── benign/
│   └── malignant/
├── validation/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

### Training the Model

```bash
python train.py --epochs 100 --batch-size 32 --learning-rate 0.001
```

**Training Options:**
```bash
--epochs          Number of training epochs (default: 100)
--batch-size      Batch size for training (default: 32)
--learning-rate   Initial learning rate (default: 0.001)
--dataset-path    Path to dataset directory
--save-path       Path to save trained model
```

### Generating Training Visualizations

After training, generate comprehensive plots:

```python
from visualize_training import visualize_from_history

# If you have history object from training
visualize_from_history(history, save_dir='results/plots/')

# Or use manual data entry
from visualize_training import visualize_from_manual_data
visualize_from_manual_data()
```

### Making Predictions

```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load trained model
model = keras.models.load_model('best_multi_dataset_cancer_model.h5')

# Load and preprocess image
img = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
confidence = prediction[0][0]

if confidence > 0.5:
    print(f"Prediction: Malignant (Confidence: {confidence:.2%})")
else:
    print(f"Prediction: Benign (Confidence: {1-confidence:.2%})")
```

---

## 📊 Detailed Model Evaluation

### Confusion Matrix

After training completion, evaluate with confusion matrix:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Get predictions
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300)
```

### Classification Report

```python
print(classification_report(y_true, y_pred_classes, 
                          target_names=['Benign', 'Malignant']))
```

Expected output format:
```
              precision    recall  f1-score   support

      Benign       0.85      0.82      0.83      4500
   Malignant       0.80      0.84      0.82      2900

    accuracy                           0.83      7400
   macro avg       0.83      0.83      0.83      7400
weighted avg       0.83      0.83      0.83      7400
```

---

## 📁 Project Structure

```
cancer-detection-model/
│
├── train.py                    # Main training script
├── predict.py                  # Inference script
├── model.py                    # Model architecture definition
├── visualize_training.py       # Visualization utilities
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── dataset/                    # Dataset directory
│   ├── train/                  # 61,876 images
│   ├── validation/             # 10,918 images
│   └── test/                   # 7,400 images
│
├── models/                     # Saved models
│   ├── best_multi_dataset_cancer_model.h5
│   ├── model_architecture.png
│   └── checkpoints/
│
├── results/                    # Training results
│   ├── plots/
│   │   ├── training_history_complete.png
│   │   ├── accuracy_plot.png
│   │   ├── loss_plot.png
│   │   └── auc_plot.png
│   └── logs/
│       └── training_log.csv
│
└── notebooks/                  # Jupyter notebooks
    ├── exploratory_analysis.ipynb
    └── model_evaluation.ipynb
```

---

## 🔧 Training on Free GPU Platforms

### Google Colab

1. Upload notebook to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Mount Google Drive for dataset access:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. Run training script
5. Save model to Drive to prevent loss

**Expected speedup:** CPU (2.5h/epoch) → GPU (2-3 min/epoch) = **50x faster**

### Kaggle Notebooks

1. Create new notebook on Kaggle
2. Add dataset or upload your own
3. Enable GPU in Settings → Accelerator
4. Verify phone number for GPU access
5. Use commit mode for background training

**GPU Quotas:**
- **Colab:** ~12 hours per session (free tier)
- **Kaggle:** 30-41 hours per week

**Your 100-epoch training:**
- CPU: ~250 hours (10+ days) ❌ Not feasible
- GPU: ~4-5 hours ✅ Recommended

---

## 🛠️ Improving Model Performance

### Current Issues & Solutions

#### Issue 1: Overfitting (Train: 87%, Val: 62%)

**Solutions:**
```python
# 1. Increase dropout
model.add(Dropout(0.5))  # Instead of 0.3

# 2. Add L2 regularization
from tensorflow.keras.regularizers import l2
Dense(128, kernel_regularizer=l2(0.01))

# 3. Early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                          restore_best_weights=True)

# 4. More aggressive data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,      # Increased from 20
    width_shift_range=0.3,  # Increased from 0.2
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]  # New
)
```

#### Issue 2: Class Imbalance

**Current handling:** Class weights applied (1.628 vs 0.722)

**Additional options:**
```python
# 1. Oversample minority class
from imblearn.over_sampling import SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

# 2. Use focal loss (focuses on hard examples)
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

model.compile(loss=focal_loss())
```

---

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Error:**
```python
# Reduce batch size
train_generator = datagen.flow_from_directory(
    batch_size=16  # Instead of 32
)
```

**Slow Training on CPU:**
```
✅ Solution: Use Google Colab or Kaggle GPU
   Expected: 50-100x speedup
   Your case: 2.5h/epoch → 2-3min/epoch
```

**Validation Accuracy Stuck:**
- Check for data leakage
- Verify validation set is representative
- Reduce model complexity
- Increase regularization

**Model Not Improving After Epoch X:**
```python
# Implement learning rate reduction
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

---

## 📚 References & Resources

### Deep Learning Architectures

- **ResNet:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Inception:** [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- **SENet:** [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- **CBAM:** [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

### Medical Imaging AI

- [Deep Learning in Medical Imaging](https://www.nature.com/articles/s41746-019-0101-y)
- [AI for Cancer Detection Review](https://www.nature.com/articles/s41568-020-0305-z)

### Handling Class Imbalance

- [SMOTE: Synthetic Minority Over-sampling](https://arxiv.org/abs/1106.1813)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 📄 License & Legal

### Copyright

```
Copyright (C) 2025-2026 Abhijeet Prabhakar
```

### License

This project is licensed under the **GNU Affero General Public License v3.0** with additional restrictions for medical use.

**ADDITIONAL RESTRICTION FOR MEDICAL/CLINICAL USE:**

This software is provided for research and educational purposes only. Any clinical, diagnostic, or commercial medical use requires:

1. Explicit written permission from the copyright holder
2. Appropriate regulatory approval (FDA, CE marking, etc.)
3. Clinical validation studies
4. Compliance with medical device regulations

### Permitted Uses

✅ Academic research  
✅ Educational purposes  
✅ Algorithm development  
✅ Non-commercial experimentation  
✅ Dataset benchmarking  

### Prohibited Uses

❌ Clinical diagnosis  
❌ Patient care decisions  
❌ Medical device integration  
❌ Commercial medical products  
❌ Healthcare screening programs  

### Liability Disclaimer

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.** The authors and copyright holders shall not be liable for any claims, damages, or liabilities arising from the use of this software, including but not limited to medical misdiagnosis, patient harm, or financial losses.

---

## 📧 Contact

### For Commercial Licensing Inquiries

**Email:** mr.robotcyber01@gmail.com

**Include in your inquiry:**
- Intended use case
- Organization details
- Expected deployment scale
- Regulatory compliance plans

### For Research Collaboration

Open to collaborations with:
- Academic institutions
- Research laboratories
- Non-profit healthcare organizations

### Reporting Issues

Please use GitHub Issues for:
- Bug reports
- Feature requests
- Documentation improvements
- Performance issues

---

## 🙏 Acknowledgments

- TensorFlow and Keras development teams
- Medical imaging dataset contributors
- Open-source deep learning community
- Researchers advancing AI in healthcare

---

## 📌 Version History

### v1.0.0 (Current)
- Initial release
- Custom CNN architecture with attention mechanisms
- Binary classification (benign/malignant)
- 2.5M parameter model
- Trained on 80K+ medical images
- Comprehensive data augmentation
- Training visualization tools

### Training Status
- ✅ Architecture defined and tested
- 🔄 Training in progress (Epoch 3/100)
- ⏳ Expected completion: ~250 hours on CPU
- 💡 Recommendation: Continue on GPU for 4-5 hours total
  
---

## ⭐ Star This Repository

If you find this project useful for your research, please consider giving it a star! It helps others discover the project and motivates continued development.

---

**Remember: This is a research tool. Always consult qualified medical professionals for health concerns.**

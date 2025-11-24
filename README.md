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
- **Total Parameters:** ~millions (computed during training)
- **Architecture Depth:** 4 main processing stages
- **Activation Functions:** ReLU, Sigmoid (output)
- **Optimization:** Adam optimizer with learning rate scheduling

---

## 📊 Dataset Information

### Training Data Sources

This model is trained on combined public medical imaging datasets:

- **Total Training Samples:** Varies by dataset configuration
- **Total Validation Samples:** 20% split from training data
- **Total Test Samples:** Separate holdout set for evaluation
- **Image Format:** RGB images (224×224 pixels)
- **Classes:** Binary (Benign vs Malignant)

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
- **Batch Size:** Configurable (default: 32)

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
python train.py --epochs 50 --batch-size 32 --learning-rate 0.001
```

**Training Options:**
```bash
--epochs          Number of training epochs (default: 50)
--batch-size      Batch size for training (default: 32)
--learning-rate   Initial learning rate (default: 0.001)
--dataset-path    Path to dataset directory
--save-path       Path to save trained model
```

### Making Predictions

```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load trained model
model = keras.models.load_model('best_model.h5')

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

## 📈 Performance Metrics

### Model Evaluation

The model is evaluated using multiple metrics:

- **Accuracy:** Overall correct predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall (Sensitivity):** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the receiver operating characteristic curve
- **Confusion Matrix:** Detailed breakdown of predictions

### Training Performance

```
Typical Training Results (Example):
├── Training Accuracy: ~92%
├── Validation Accuracy: ~88%
├── Test Accuracy: ~85%
└── Training Time: ~2-3 hours (GPU)
```

### Visualization Tools

The project includes scripts to visualize:

- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrix heatmap
- ROC curve with AUC score
- Sample predictions with confidence scores

---

## 📁 Project Structure

```
cancer-detection-model/
│
├── train.py                 # Main training script
├── predict.py              # Inference script
├── model.py                # Model architecture definition
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── dataset/               # Dataset directory
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/                # Saved models
│   ├── best_model.h5
│   └── checkpoints/
│
├── results/               # Training results
│   ├── plots/
│   └── logs/
│
└── notebooks/             # Jupyter notebooks
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

### Kaggle Notebooks

1. Create new notebook on Kaggle
2. Add dataset or upload your own
3. Enable GPU in Settings → Accelerator
4. Verify phone number for GPU access
5. Use commit mode for background training

**GPU Quotas:**
- **Colab:** ~12 hours per session (free tier)
- **Kaggle:** 30-41 hours per week

---

## 🛠️ Improving Model Performance

### Techniques to Try

1. **Increase Training Data:**
   - Add more diverse samples
   - Use more aggressive augmentation
   - Collect from multiple datasets

2. **Architecture Modifications:**
   - Add more residual blocks
   - Increase model depth
   - Experiment with different attention mechanisms

3. **Hyperparameter Tuning:**
   - Learning rate scheduling
   - Different optimizers (Adam, SGD, RMSprop)
   - Batch size optimization

4. **Regularization:**
   - Increase dropout rates
   - Add L2 regularization
   - Use batch normalization

5. **Transfer Learning:**
   - Use pre-trained models (ResNet, EfficientNet)
   - Fine-tune on your dataset

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

**Low Accuracy (~50-60%):**
- Check data quality and labeling
- Increase training epochs
- Verify data augmentation isn't too aggressive
- Check for class imbalance

**Model Not Learning:**
- Reduce learning rate
- Check loss function is appropriate
- Verify data preprocessing
- Ensure labels are correct

**Session Timeout (Colab/Kaggle):**
```python
# Save checkpoints every N epochs
checkpoint = ModelCheckpoint(
    'checkpoint_epoch_{epoch}.h5',
    save_freq='epoch'
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

### Datasets (Examples)

- NIH Chest X-ray Dataset
- ISIC Skin Lesion Dataset
- BreakHis Breast Cancer Histopathology
- **Note:** Verify licensing before use

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
- Comprehensive data augmentation
- Training scripts and evaluation tools

---

## ⭐ Star This Repository

If you find this project useful for your research, please consider giving it a star! It helps others discover the project and motivates continued development.

---

**Remember: This is a research tool. Always consult qualified medical professionals for health concerns.**

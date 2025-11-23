# Copyright (C) 2025-2026 [Abhijeet Prabhakar]
### For research use only. Clinical use prohibited without authorization.

This cancer detection AI model is licensed under GNU Affero General Public License v3.0
with the following additional restriction:
# 🏥 Multi-Dataset Cancer Detection Model

## ⚠️ CRITICAL WARNING - RESEARCH USE ONLY

**THIS MODEL IS NOT APPROVED FOR CLINICAL USE**

This model is intended for **research and educational purposes ONLY**. It has NOT been:
- ✗ Validated in clinical settings
- ✗ Approved by FDA, EMA, or other regulatory bodies  
- ✗ Tested for safety in medical decision-making
- ✗ Peer-reviewed or published

**DO NOT USE for:**
- Clinical diagnosis
- Patient care decisions
- Medical screening
- Commercial medical applications
- Any life-critical decisions

## 📋 Model Description

This is an advanced custom Convolutional Neural Network built from scratch for cancer detection, trained on multiple public medical imaging datasets. The model uses:

- **Advanced Architecture Features:**
  - Residual connections with Squeeze-and-Excitation blocks
  - Multi-scale feature extraction (Inception-style modules)
  - Channel and Spatial Attention (CBAM)
  - Global Average Pooling to reduce overfitting
  - Deep classification head with dropout regularization

- **Training Data:** Combined from multiple public datasets
  - Total Training Samples: {train_generator.samples}
  - Total Validation Samples: {validation_generator.samples}
  - Total Test Samples: {test_generator.samples}

- **Model Architecture:**
  - Input: 224x224x3 RGB images
  - Total Parameters: {model.count_params():,}
  - Depth: Multi-stage CNN with 4 main stages
  - Output: Binary classification (benign vs malignant)

ADDITIONAL RESTRICTION FOR MEDICAL/CLINICAL USE:
This software is provided for research and educational purposes only. 
Any clinical, diagnostic, or commercial medical use requires explicit 
written permission from the copyright holder and appropriate regulatory 
approval (FDA, CE marking, etc.).

For commercial licensing inquiries, contact: [mr.robotcyber01@gmail.com]

## ⚠️ Important Notice
### This is a research project. NOT approved for clinical diagnosis.
### Not FDA approved. Not intended for medical decision-making.     

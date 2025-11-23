"""
05_evaluate_model.py
Evaluate the trained model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model_path='best_multi_dataset_cancer_model.h5', data_dir='combined_data'):
    """Comprehensive model evaluation"""
    
    print("🔍 Loading model for evaluation...")
    model = keras.models.load_model(model_path)
    
    # Load test data
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        f'{data_dir}/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\n📊 Evaluating on test set...")
    results = model.evaluate(test_generator, verbose=1)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Loss:      {results[0]:.4f}")
    print(f"Accuracy:  {results[1]:.4f} ({results[1]*100:.2f}%)")
    print(f"AUC:       {results[2]:.4f}")
    print(f"Precision: {results[3]:.4f}")
    print(f"Recall:    {results[4]:.4f}")
    print(f"F1 Score:  {2 * (results[3] * results[4]) / (results[3] + results[4]):.4f}")
    
    # Predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    # Classification report
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(true_classes, predicted_classes, 
                                target_names=class_labels, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_multi_dataset.png', dpi=300)
    plt.show()
    print("✅ Confusion matrix saved!")
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve((true_classes == i).astype(int), predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves_multi_dataset.png', dpi=300)
        plt.show()
    print("✅ ROC curves saved!")
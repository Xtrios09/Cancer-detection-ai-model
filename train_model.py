"""
04_train_model.py
Train the cancer detection model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from model_architecture import create_cancer_detection_model

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 100
DATA_DIR = 'combined_data'

def create_data_generators():
    """Create data generators with augmentation"""
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20,
        fill_mode='reflect',
        validation_split=0.15
    )
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        f'{DATA_DIR}/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        f'{DATA_DIR}/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        f'{DATA_DIR}/test',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def train_model():
    """Main training function"""
    
    print("🚀 Starting Cancer Detection Model Training")
    print("=" * 60)
    
    # Create data generators
    print("\n📂 Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n⚖️ Class weights: {class_weight_dict}")
    
    # Create model
    print("\n🏗️ Building model...")
    model = create_cancer_detection_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=len(train_gen.class_indices)
    )
    
    # Learning rate schedule
    initial_lr = 0.001
    decay_steps = len(train_gen) * 50
    lr_schedule = CosineDecay(initial_lr, decay_steps, alpha=0.0)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Callbacks
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_multi_dataset_cancer_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger('training_history.csv'),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\n📊 Epoch {epoch+1}:"
                f"\n  Acc: {logs['accuracy']:.4f} | Val Acc: {logs['val_accuracy']:.4f}"
                f"\n  AUC: {logs['auc']:.4f} | Val AUC: {logs['val_auc']:.4f}"
            )
        )
    ]
    
    # Train
    print(f"\n🎯 Training for {EPOCHS} epochs...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save final model
    model.save('final_multi_dataset_cancer_model.h5')
    model.save('final_multi_dataset_cancer_model')
    
    print("\n✅ Training complete!")
    print("✅ Models saved successfully!")
    
    return model, history, test_gen

if __name__ == "__main__":
    model, history, test_gen = train_model()
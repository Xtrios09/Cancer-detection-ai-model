"""
03_model_architecture.py
Define advanced custom CNN architecture for cancer detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def residual_se_block(x, filters, blocks, name):
    """Residual block with Squeeze-and-Excitation"""
    for i in range(blocks):
        shortcut = x
        
        # Bottleneck
        x = layers.Conv2D(filters // 4, 1, padding='same', name=f'{name}_conv{i+1}a')(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i+1}a')(x)
        x = layers.Activation('relu', name=f'{name}_relu{i+1}a')(x)
        
        x = layers.Conv2D(filters // 4, 3, padding='same', name=f'{name}_conv{i+1}b')(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i+1}b')(x)
        x = layers.Activation('relu', name=f'{name}_relu{i+1}b')(x)
        
        x = layers.Conv2D(filters, 1, padding='same', name=f'{name}_conv{i+1}c')(x)
        x = layers.BatchNormalization(name=f'{name}_bn{i+1}c')(x)
        
        # SE block
        se = layers.GlobalAveragePooling2D(name=f'{name}_se{i+1}_gap')(x)
        se = layers.Dense(filters // 16, activation='relu', name=f'{name}_se{i+1}_fc1')(se)
        se = layers.Dense(filters, activation='sigmoid', name=f'{name}_se{i+1}_fc2')(se)
        se = layers.Reshape((1, 1, filters), name=f'{name}_se{i+1}_reshape')(se)
        x = layers.Multiply(name=f'{name}_se{i+1}_multiply')([x, se])
        
        # Shortcut
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same', name=f'{name}_shortcut{i+1}')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn{i+1}')(shortcut)
        
        x = layers.Add(name=f'{name}_add{i+1}')([x, shortcut])
        x = layers.Activation('relu', name=f'{name}_relu{i+1}c')(x)
    
    return x

def multi_scale_block(x, filters, name):
    """Multi-scale feature extraction"""
    # 1x1
    branch1 = layers.Conv2D(filters // 4, 1, padding='same', name=f'{name}_b1_conv')(x)
    branch1 = layers.BatchNormalization(name=f'{name}_b1_bn')(branch1)
    branch1 = layers.Activation('relu', name=f'{name}_b1_relu')(branch1)
    
    # 3x3
    branch2 = layers.Conv2D(filters // 4, 1, padding='same', name=f'{name}_b2_conv1')(x)
    branch2 = layers.BatchNormalization(name=f'{name}_b2_bn1')(branch2)
    branch2 = layers.Activation('relu', name=f'{name}_b2_relu1')(branch2)
    branch2 = layers.Conv2D(filters // 4, 3, padding='same', name=f'{name}_b2_conv2')(branch2)
    branch2 = layers.BatchNormalization(name=f'{name}_b2_bn2')(branch2)
    branch2 = layers.Activation('relu', name=f'{name}_b2_relu2')(branch2)
    
    # 5x5
    branch3 = layers.Conv2D(filters // 4, 1, padding='same', name=f'{name}_b3_conv1')(x)
    branch3 = layers.BatchNormalization(name=f'{name}_b3_bn1')(branch3)
    branch3 = layers.Activation('relu', name=f'{name}_b3_relu1')(branch3)
    branch3 = layers.Conv2D(filters // 4, 3, padding='same', name=f'{name}_b3_conv2')(branch3)
    branch3 = layers.BatchNormalization(name=f'{name}_b3_bn2')(branch3)
    branch3 = layers.Activation('relu', name=f'{name}_b3_relu2')(branch3)
    branch3 = layers.Conv2D(filters // 4, 3, padding='same', name=f'{name}_b3_conv3')(branch3)
    branch3 = layers.BatchNormalization(name=f'{name}_b3_bn3')(branch3)
    branch3 = layers.Activation('relu', name=f'{name}_b3_relu3')(branch3)
    
    # Pooling
    branch4 = layers.MaxPooling2D(3, strides=1, padding='same', name=f'{name}_b4_pool')(x)
    branch4 = layers.Conv2D(filters // 4, 1, padding='same', name=f'{name}_b4_conv')(branch4)
    branch4 = layers.BatchNormalization(name=f'{name}_b4_bn')(branch4)
    branch4 = layers.Activation('relu', name=f'{name}_b4_relu')(branch4)
    
    return layers.Concatenate(axis=-1, name=f'{name}_concat')([branch1, branch2, branch3, branch4])

def attention_block(x, filters, name):
    """CBAM: Channel and Spatial Attention"""
    # Channel attention
    avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_ca_avg')(x)
    avg_pool = layers.Reshape((1, 1, filters), name=f'{name}_ca_reshape1')(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D(name=f'{name}_ca_max')(x)
    max_pool = layers.Reshape((1, 1, filters), name=f'{name}_ca_reshape2')(max_pool)
    
    dense1 = layers.Dense(filters // 8, activation='relu', name=f'{name}_ca_fc1')
    dense2 = layers.Dense(filters, name=f'{name}_ca_fc2')
    
    avg_pool = dense2(dense1(avg_pool))
    max_pool = dense2(dense1(max_pool))
    
    channel_att = layers.Add(name=f'{name}_ca_add')([avg_pool, max_pool])
    channel_att = layers.Activation('sigmoid', name=f'{name}_ca_sigmoid')(channel_att)
    x = layers.Multiply(name=f'{name}_ca_multiply')([x, channel_att])
    
    # Spatial attention
    avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1, name=f'{name}_sa_concat')([avg_spatial, max_spatial])
    
    spatial_att = layers.Conv2D(1, 7, padding='same', name=f'{name}_sa_conv')(concat)
    spatial_att = layers.Activation('sigmoid', name=f'{name}_sa_sigmoid')(spatial_att)
    x = layers.Multiply(name=f'{name}_sa_multiply')([x, spatial_att])
    
    return x

def create_cancer_detection_model(input_shape=(224, 224, 3), num_classes=2):
    """Create advanced cancer detection model"""
    
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='stem_pool')(x)
    
    # Stage 1
    x = residual_se_block(x, filters=64, blocks=3, name='stage1')
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Stage 2
    x = residual_se_block(x, filters=128, blocks=4, name='stage2')
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    # Stage 3 - Multi-scale
    x = multi_scale_block(x, filters=256, name='stage3')
    x = layers.MaxPooling2D(2, name='pool3')(x)
    
    # Stage 4 - with Attention
    x = residual_se_block(x, filters=512, blocks=3, name='stage4')
    x = attention_block(x, filters=512, name='attention')
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    x = layers.Dense(1024, name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Activation('relu', name='fc1_relu')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    
    x = layers.Dense(512, name='fc2')(x)
    x = layers.BatchNormalization(name='fc2_bn')(x)
    x = layers.Activation('relu', name='fc2_relu')(x)
    x = layers.Dropout(0.4, name='dropout2')(x)
    
    x = layers.Dense(256, name='fc3')(x)
    x = layers.BatchNormalization(name='fc3_bn')(x)
    x = layers.Activation('relu', name='fc3_relu')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='CancerDetectionCNN')
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_cancer_detection_model()
    model.summary()
    
    # Save architecture diagram
    keras.utils.plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=96
    )
    print("\n✅ Model architecture saved to 'model_architecture.png'")
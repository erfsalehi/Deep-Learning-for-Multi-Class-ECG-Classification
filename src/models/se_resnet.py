import tensorflow as tf
from tensorflow.keras import layers, models

def squeeze_excitation_block(input_tensor, ratio=16):
    """
    Squeeze and Excitation Block
    """
    filters = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling1D()(input_tensor)
    
    # Excitation: Dense -> ReLU -> Dense -> Sigmoid
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    
    # Reshape to match input dimensions
    se = layers.Reshape((1, filters))(se)
    
    # Scale input
    x = layers.Multiply()([input_tensor, se])
    return x

def residual_block_se(x, filters, kernel_size=5, stride=1):
    """
    Residual Block with SE
    """
    shortcut = x
    
    # First Conv
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Second Conv
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # SE Block
    x = squeeze_excitation_block(x)
    
    # Shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def create_se_resnet(input_shape=(1000, 12), num_classes=5):
    """
    SE-ResNet Model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Blocks
    x = residual_block_se(x, 32, kernel_size=5)
    x = residual_block_se(x, 64, kernel_size=5, stride=2)
    x = residual_block_se(x, 64, kernel_size=5)
    x = residual_block_se(x, 128, kernel_size=3, stride=2)
    x = residual_block_se(x, 128, kernel_size=3)
    x = residual_block_se(x, 256, kernel_size=3, stride=2)
    x = residual_block_se(x, 256, kernel_size=3)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="SE_ResNet")
    return model

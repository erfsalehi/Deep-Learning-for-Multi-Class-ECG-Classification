import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(x, filters, kernel_size=5, stride=1):
    """
    A residual block with two 1D convolutions and a skip connection.
    """
    shortcut = x
    
    # First convolution
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x) # Add dropout for regularization

    # Second convolution
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if dimensions don't match
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_resnet(input_shape=(1000, 12), num_classes=2):
    """
    Builds a ResNet-style 1D-CNN for ECG classification.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv
    x = layers.Conv1D(32, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual Blocks
    # We increase filters and downsample (stride=2) periodically
    x = residual_block(x, 32, kernel_size=5)
    x = residual_block(x, 64, kernel_size=5, stride=2)
    x = residual_block(x, 64, kernel_size=5)
    x = residual_block(x, 128, kernel_size=3, stride=2)
    x = residual_block(x, 128, kernel_size=3)
    x = residual_block(x, 256, kernel_size=3, stride=2)
    x = residual_block(x, 256, kernel_size=3)
    
    # Global Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_ECG")
    return model

if __name__ == "__main__":
    model = create_resnet()
    model.summary()

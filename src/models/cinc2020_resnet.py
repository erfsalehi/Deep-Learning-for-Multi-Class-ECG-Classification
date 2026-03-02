import tensorflow as tf
from tensorflow.keras import layers, models

def _cinc_block(x, filters, kernel_size, downsample=False):
    shortcut = x
    strides = 2 if downsample else 1
    
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def build_cinc2020_resnet(input_shape=(1000, 12), num_classes=5):
    """
    A Wide 1D ResNet facsimile for the PhysioNet/CinC 2020 competition winners.
    Uses wider layers and deeper bottleneck blocks.
    """
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(64, 7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 3 Stages of residual blocks with increasing width
    x = _cinc_block(x, 128, 3, downsample=True)  # -> 500
    x = _cinc_block(x, 128, 3)
    
    x = _cinc_block(x, 256, 3, downsample=True)  # -> 250
    x = _cinc_block(x, 256, 3)
    
    x = _cinc_block(x, 512, 3, downsample=True)  # -> 125
    x = _cinc_block(x, 512, 3)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='cinc2020_resnet')
    return model

if __name__ == '__main__':
    model = build_cinc2020_resnet()
    model.summary()

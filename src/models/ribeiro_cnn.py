import tensorflow as tf
from tensorflow.keras import layers, models

def _residual_block(x, filters, kernel_size, downsample=False):
    """
    Standard ResNet 1D block used in Ribeiro et al. (2020).
    """
    shortcut = x

    # Path A
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    
    # Path B: strided convolution for downsampling
    strides = 2 if downsample else 1
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut matching
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=strides, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def build_ribeiro_resnet(input_shape=(1000, 12), num_classes=5):
    """
    Builds the 1D ResNet architecture proposed by Ribeiro et al. (2020),
    adapted for short sequences if necessary.
    Original paper uses (4096, 12) inputs and four residual blocks.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv1D(64, kernel_size=15, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Four residual blocks
    x = _residual_block(x, filters=128, kernel_size=15, downsample=True)  # -> 500
    x = _residual_block(x, filters=192, kernel_size=15, downsample=True)  # -> 250
    x = _residual_block(x, filters=256, kernel_size=15, downsample=True)  # -> 125
    x = _residual_block(x, filters=320, kernel_size=15, downsample=True)  # -> 62
    
    # Global Pooling and Output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ribeiro_resnet_1d')
    return model

if __name__ == '__main__':
    model = build_ribeiro_resnet()
    model.summary()

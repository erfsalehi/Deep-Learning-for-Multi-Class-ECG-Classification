import tensorflow as tf
from tensorflow.keras import layers, models

def create_1d_cnn(input_shape=(1000, 12), num_classes=2):
    """
    Creates a simple 1D-CNN model suitable for laptop training.
    """
    model = models.Sequential()
    
    # 1st Conv Block
    model.add(layers.Conv1D(32, kernel_size=7, activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # 2nd Conv Block
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # 3rd Conv Block
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling1D())
    
    # Dense Layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    model = create_1d_cnn()
    model.summary()

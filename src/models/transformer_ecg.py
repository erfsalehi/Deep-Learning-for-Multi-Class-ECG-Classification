import tensorflow as tf
from tensorflow.keras import layers, models

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Dense(embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        })
        return config

    def call(self, x):
        # x shape: (batch, seq_len, channels)
        # divide into patches
        batch_size = tf.shape(x)[0]
        num_patches = x.shape[1] // self.patch_size
        x = tf.reshape(x, (batch_size, num_patches, self.patch_size * x.shape[2]))
        return self.projection(x)

def build_transformer_ecg(input_shape=(1000, 12), num_classes=5, patch_size=50, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4):
    """
    Builds a simple 1D Transformer (Vision Transformer style) for ECG classification.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Patch Embedding
    x = PatchEmbedding(patch_size, embed_dim)(inputs)
    
    # Positional Encoding (Learnable)
    num_patches = input_shape[0] // patch_size
    pos_emb = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(tf.range(start=0, limit=num_patches, delta=1))
    x = x + pos_emb
    
    # Transformer Blocks
    for _ in range(num_layers):
        # Attention
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed Forward
        ff_out = layers.Dense(ff_dim, activation='relu')(x)
        ff_out = layers.Dense(embed_dim)(ff_out)
        x = layers.Add()([x, ff_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
    # Global Pooling and Output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='transformer_ecg')
    return model

if __name__ == '__main__':
    model = build_transformer_ecg()
    model.summary()

import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_multiscale_cnn(input_shape=(32, 32, 3), num_classes=43):
    input_layer = Input(shape=input_shape)

    # First scale (original resolution)
    x1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = layers.Flatten()(x1)

    # Second scale (downsampled)
    x2 = layers.AveragePooling2D(pool_size=(2, 2))(input_layer)
    x2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x2)
    x2 = layers.Flatten()(x2)

    # Merge both scales
    merged = layers.concatenate([x1, x2])
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    output = layers.Dense(num_classes, activation='softmax')(merged)

    model = models.Model(inputs=input_layer, outputs=output)
    return model
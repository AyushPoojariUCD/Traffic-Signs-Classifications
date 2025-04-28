import tensorflow as tf
from tensorflow.keras import layers, models

# MCDNN Model
def build_mcdnn_column(input_shape=(32, 32, 3), num_classes=43):
    model = models.Sequential()

    model.add(layers.Conv2D(100, (7, 7), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(150, (4, 4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(250, (4, 4), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(300, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Function to create an average ensemble of multiple MCDNN columns
def average_ensemble(models_list, input_shape=(32, 32, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = [model(inputs) for model in models_list]
    avg_output = layers.Average()(outputs)
    return tf.keras.Model(inputs=inputs, outputs=avg_output)

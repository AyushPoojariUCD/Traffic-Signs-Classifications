from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

def build_alexnet(input_shape=(32, 32, 3), num_classes=43):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(filters=96, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

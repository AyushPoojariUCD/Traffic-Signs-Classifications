from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, InputLayer

def build_lenet(input_shape=(32, 32, 3), num_classes=43):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

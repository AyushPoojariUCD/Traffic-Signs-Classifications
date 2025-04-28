import numpy as np
from tensorflow.keras.utils import to_categorical

# Data Preprocess Function
def preprocess_data(data,num_classes = 43, verbose = True):

    # Convert labels to one-hot and float32
    data['y_train'] = to_categorical(data['y_train'], num_classes=num_classes).astype('float32')
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=num_classes).astype('float32')

    # Transpose to (samples, height, width, channels)
    for key in ['x_train', 'x_validation', 'x_test']:
        data[key] = data[key].transpose(0, 2, 3, 1)

        # Clamp values between 0 and 255, convert to float32 and normalize
        data[key] = np.clip(data[key], 0, 255).astype('float32') / 255.0

    if verbose:
        for key, value in data.items():
            if key == 'labels':
                print(f"{key}: {len(value)}")
            else:
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")

    #print("x_train max:", data['x_train'].max())
    #print("x_train min:", data['x_train'].min())
    #print("y_train[0]:", data['y_train'][0])

    return data




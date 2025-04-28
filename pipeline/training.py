import time
from models.cnn import build_cnn_model
from models.lenet import build_lenet
from models.alexnet import build_alexnet
from models.vgg import build_vgg
from models.resnet import build_resnet
from models.multiscale import build_multiscale_cnn
from models.mcdnn import build_mcdnn_column, average_ensemble
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

# Training Model
def train_model(data, model_type, epochs=10, batch_size=128):
    # Model selection based on the provided model type
    if model_type == "cnn":
        model = build_cnn_model(input_shape=(32, 32, 3), num_classes=43)
    elif model_type == "lenet":
        model = build_lenet(input_shape=(32, 32, 3), num_classes=43)
    elif model_type == "alexnet":
        model = build_alexnet(input_shape=(32, 32, 3), num_classes=43)
    elif model_type == "vgg":
        model = build_vgg(input_shape=(32, 32, 3), num_classes=43)
    elif model_type == "resnet":
        model = build_resnet(input_shape=(32, 32, 3), num_classes=43)
    elif model_type == "multiscale":
        model = build_multiscale_cnn(input_shape=(32, 32, 3), num_classes=43)
    elif model_type == "mcdnn":
        columns = [build_mcdnn_column(input_shape=(32, 32, 3), num_classes=43) for _ in range(5)]
        for mcdnn_model in columns:
            mcdnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model = average_ensemble(columns)
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='max', restore_best_weights=True)
    ]

    # Start training time tracking
    start_time = time.time()

    # Model training
    history = model.fit(
        data['x_train'], data['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(data['x_validation'], data['y_validation']),
        callbacks=callbacks,
        verbose=1
    )

    # End training time tracking
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time:.2f} seconds")


    # Saving the model
    os.makedirs('./artifacts', exist_ok=True)
    model.save(f'./artifacts/{model_type}_model.h5',save_format='h5')
    print(f"{model_type} model saved in .keras format")

    # Saving training time separately
    os.makedirs('./logs', exist_ok=True)
    with open(f'./logs/{model_type}_training_time.txt', 'w') as f:
        f.write(str(training_time))

    return model, history, training_time



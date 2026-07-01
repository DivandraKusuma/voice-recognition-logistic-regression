import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def build_model(input_shape):

    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Fully Connected
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output layer (binary classification)
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=32):

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate_model(model, X_test, y_test):

    # Prediksi probabilitas dan konversi ke label biner
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_int = y_test.astype(int).flatten()

    results = {
        'accuracy': accuracy_score(y_test_int, y_pred),
        'precision': precision_score(y_test_int, y_pred, average='binary'),
        'recall': recall_score(y_test_int, y_pred, average='binary'),
        'f1': f1_score(y_test_int, y_pred, average='binary'),
        'classification_report': classification_report(
            y_test_int, y_pred, target_names=['female', 'male']
        ),
        'confusion_matrix': confusion_matrix(y_test_int, y_pred),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }

    return results

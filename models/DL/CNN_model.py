import tensorflow as tf
from tensorflow.keras import layers, models

def get_cnn_model(input_shape=(128, 130, 1)):
    # Membuat cetakan arsitektur CNN untuk klasifikasi gambar spektrogram
    model = models.Sequential([
        # Layer Konvolusi 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Layer Konvolusi 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening (Mengubah matriks jadi vektor 1D)
        layers.Flatten(),
        
        # Fully Connected Layer (Dense)
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Mencegah overfitting
        
        # Output Layer (1 neuron untuk Binary Classification: Male/Female)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model dengan optimizer Adam dan loss binary crossentropy
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
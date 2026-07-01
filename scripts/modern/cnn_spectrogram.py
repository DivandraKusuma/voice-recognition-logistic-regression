import numpy as np
from sklearn.model_selection import train_test_split
from models.DL.CNN_model import get_cnn_model

def run_cnn_training(x_path="x_spectrogram_data.npy", y_path="y_gender_labels.npy"):
    print("\nCNN SPECTROGRAM TRAINING")
    
    X = np.load(x_path)
    y = np.load(y_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    input_shape = X_train.shape[1:] 
    model = get_cnn_model(input_shape=input_shape)
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=10, 
                        batch_size=32)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Akurasi Akhir CNN pada Data Test: {accuracy:.4f}")
    
    return {'CNN': accuracy, 'history': history.history}
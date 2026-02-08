import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Import channel simulation logic
import channel_sim

def generate_synthetic_data(num_samples=1000, seq_len=100):
    """
    Generates training data: clean signals (sine + bits) and their noisy versions.
    """
    X_clean = []
    X_noisy = []
    
    t = np.linspace(0, 10, seq_len)
    
    for _ in range(num_samples):
        # Sine wave with random phase and frequency
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        sine = np.sin(2 * np.pi * freq * t + phase)
        
        # Random bits
        bits = np.zeros(seq_len)
        num_bits = np.random.randint(1, 5)
        for _ in range(num_bits):
            start = np.random.randint(0, seq_len - 10)
            bits[start:start+10] = 0.5 if np.random.rand() > 0.5 else 0
            
        clean = sine + bits
        
        # Add Acoustic Effects: Multipath (Echo) -> Noise -> Packet Loss
        echoed = channel_sim.apply_multipath(clean, delay=3, attenuation=0.4)
        
        snr = np.random.uniform(10, 25)
        noisy = channel_sim.add_awgn(echoed, snr_db=snr)
        
        loss_rate = np.random.uniform(0.05, 0.2)
        with_loss = channel_sim.apply_packet_loss(noisy, loss_rate=loss_rate)
        restored = channel_sim.handle_missing_data(with_loss)
        
        X_clean.append(clean)
        X_noisy.append(restored)
        
    return np.array(X_noisy)[..., np.newaxis], np.array(X_clean)[..., np.newaxis]

def build_autoencoder(seq_len=100):
    """
    Builds a 1D CNN Autoencoder for signal denoising.
    """
    input_layer = layers.Input(shape=(seq_len, 1))
    
    # Encoder
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Decoder
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    output_layer = layers.Conv1D(1, 3, activation='linear', padding='same')(x)
    
    # Note: If seq_len is not divisible by max pooling factors (4 here), 
    # the output might have a different length. We crop or pad to fix.
    if output_layer.shape[1] != seq_len:
        output_layer = layers.Resizing(seq_len, 1)(output_layer) # For fixed size output

    model = models.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    SEQ_LEN = 100
    
    print("Generating data...")
    X_train_noisy, X_train_clean = generate_synthetic_data(2000, SEQ_LEN)
    X_val_noisy, X_val_clean = generate_synthetic_data(200, SEQ_LEN)
    
    print("Building model...")
    model = build_autoencoder(SEQ_LEN)
    model.summary()
    
    print("Training model...")
    history = model.fit(
        X_train_noisy, X_train_clean,
        epochs=30,
        batch_size=32,
        validation_data=(X_val_noisy, X_val_clean),
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'equalizer_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('training_history.png')
    print("Training plot saved to training_history.png")

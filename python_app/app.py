import os
import time
import threading
import numpy as np
import serial
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import tensorflow as tf

# Import internal modules
import channel_sim

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configuration ---
SERIAL_PORT = 'COM5' # Change to your ESP32 port
BAUD_RATE = 115200
SEQ_LEN = 100
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'equalizer_model.h5')

# Status State
state = {
    'connected': False,
    'using_mock': False
}

# Load Model
model = None
if os.path.exists(MODEL_PATH):
    print("Loading DL Model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: Model file not found. Run model_training.py first.")

class MockSerial:
    """Simulates Serial data if hardware is not connected."""
    def __init__(self):
        self.t = 0
    def readline(self):
        time.sleep(0.02) # 50Hz
        sine = np.sin(2 * np.pi * 1.0 * self.t)
        bits = 0.5 if (int(self.t * 10) % 20 > 10) else 0
        val = sine + bits
        self.t += 0.02
        return f"{val:.4f}\n".encode()
    def close(self):
        pass

def serial_reader_thread():
    """Background thread to read data and perform inference with auto-reconnect."""
    global state
    
    ser = None
    mock_ser = MockSerial()
    data_window = []
    
    while True:
        # Try to connect if not connected
        if not state['connected']:
            try:
                if ser: ser.close()
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
                state['connected'] = True
                state['using_mock'] = False
                print(f"Connected to ESP32 on {SERIAL_PORT}")
            except serial.SerialException as e:
                err_msg = str(e)
                if "PermissionError" in err_msg or "Access is denied" in err_msg:
                    print(f"--- SERIAL ERROR ---")
                    print(f"Port {SERIAL_PORT} is BUSY.")
                    print(f"1. Close Arduino Serial Monitor/IDE.")
                    print(f"2. Ensure NO OTHER background ‘app.py’ is running.")
                    print(f"---------------------")
                state['connected'] = False
                state['using_mock'] = True
                current_ser = mock_ser
                time.sleep(2) # Wait a bit before retrying
            except Exception:
                state['connected'] = False
                state['using_mock'] = True
                current_ser = mock_ser
        else:
            current_ser = ser

        try:
            line = current_ser.readline().decode().strip()
            if line:
                try:
                    clean_val = float(line)
                    data_window.append(clean_val)
                    
                    # Ensure buffer doesn't grow indefinitely if processing fails
                    if len(data_window) > SEQ_LEN:
                        data_window = data_window[-SEQ_LEN:]
                    
                    if len(data_window) == SEQ_LEN:
                        window_arr = np.array(data_window)
                        
                        # Simulate Acoustic Channel: Multipath (Echo) -> Noise -> Packet Loss
                        echo_window = channel_sim.apply_multipath(window_arr, delay=3, attenuation=0.4)
                        noisy_window = channel_sim.add_awgn(echo_window, snr_db=15)
                        with_loss = channel_sim.apply_packet_loss(noisy_window, loss_rate=0.1)
                        
                        input_data = channel_sim.handle_missing_data(with_loss)
                        
                        if model:
                            input_tensor = input_data.reshape(1, SEQ_LEN, 1)
                            # model.predict returns a numpy array, we need native floats for JSON
                            reconstructed = model.predict(input_tensor, verbose=0).flatten()
                        else:
                            reconstructed = input_data
                        
                        mse = float(np.mean((window_arr - reconstructed)**2))
                        
                        socketio.emit('new_data', {
                            'clean': float(window_arr[-1]),
                            'noisy': float(with_loss[-1]) if not np.isnan(with_loss[-1]) else None,
                            'reconstructed': float(reconstructed[-1]),
                            'mse': mse,
                            'status': state
                        })
                        
                        # Slide window
                        data_window = data_window[1:]
                except ValueError:
                    continue
            
            if state['connected'] and not line:
                if not ser.is_open:
                    state['connected'] = False
                    
        except Exception as e:
            # print(f"Communication error: {e}") # Suppress to avoid spam, but keep for debug if needed
            state['connected'] = False
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Start serial thread
    thread = threading.Thread(target=serial_reader_thread, daemon=True)
    thread.start()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

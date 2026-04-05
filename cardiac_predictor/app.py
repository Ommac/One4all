"""
Flask API for cardiac arrhythmia prediction.
Provides endpoint for ECG signal classification using CNN and LSTM models.
"""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.preprocessing import butterworth_filter, load_dataset
from src.model_definitions import ECGCNNModel as ECGNet, ECGLSTMModel as ECGLSTMNet
from src.ensemble import ensemble_predict


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model references
cnn_model = None
lstm_model = None
device = None

# Label mapping (alphabetical order from LabelEncoder)
LABEL_MAP = {0: "AFib", 1: "Normal", 2: "VFib"}


def load_models():
    """
    Load CNN and LSTM models at startup.
    Called once when the application starts.
    """
    global cnn_model, lstm_model, device
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Signal parameters (must match training)
    SIGNAL_LENGTH = 2500  # Original signal length
    LSTM_LENGTH = 500     # Downsampled for LSTM (every 5th point)
    
    # Load CNN model
    print("Loading CNN model...")
    cnn_model = ECGNet(input_length=SIGNAL_LENGTH)
    cnn_model.load_state_dict(torch.load('models/cnn_model.h5', map_location=device, weights_only=True))
    cnn_model.to(device)
    cnn_model.eval()
    
    # Load LSTM model
    print("Loading LSTM model...")
    lstm_model = ECGLSTMNet(input_length=LSTM_LENGTH)
    lstm_model.load_state_dict(torch.load('models/lstm_model.h5', map_location=device, weights_only=True))
    lstm_model.to(device)
    lstm_model.eval()
    
    print("All models loaded successfully!")


def parse_signal_from_csv(file_content: bytes) -> np.ndarray:
    """
    Parse ECG signal from uploaded CSV file.
    
    Handles multiple formats:
    1. Full dataset format with 'label' and 'signal' columns
    2. Raw signal as single column of values
    3. Signal spread across multiple columns
    
    Args:
        file_content: Raw bytes of uploaded CSV file
        
    Returns:
        1D numpy array of signal values (typically 2500 points)
    """
    EXPECTED_LENGTH = 2500
    
    try:
        # Read CSV
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Check if it's dataset format with 'label' column
        if 'label' in df.columns:
            # Use load_dataset logic for first row
            X, _ = load_dataset_from_bytes(file_content)
            return X[0]
        
        # Check if single column of signal values
        if len(df.columns) == 1:
            signal = df.iloc[:, 0].values.astype(np.float32)
            if len(signal) >= EXPECTED_LENGTH:
                return signal[:EXPECTED_LENGTH]
        
        # Try reading as spread across columns (single row)
        if len(df) == 1:
            signal = df.iloc[0].values.astype(np.float32)
            if len(signal) >= EXPECTED_LENGTH:
                return signal[:EXPECTED_LENGTH]
        
        # Try reading all values as signal
        signal = df.values.flatten().astype(np.float32)
        if len(signal) >= EXPECTED_LENGTH:
            return signal[:EXPECTED_LENGTH]
        
        raise ValueError(f"Could not parse signal. Expected at least {EXPECTED_LENGTH} values, got {len(signal)}")
        
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")


def load_dataset_from_bytes(file_content: bytes) -> tuple:
    """
    Load dataset from bytes (for uploaded files).
    
    Reuses logic from preprocessing.load_dataset but works with bytes.
    
    Args:
        file_content: Raw bytes of CSV file
        
    Returns:
        Tuple of (X, y) arrays
    """
    import ast
    import json
    
    label_map = {"Normal": 0, "AFib": 1, "VFib": 2}
    
    df = pd.read_csv(io.BytesIO(file_content))
    
    signals = []
    labels = []
    
    for idx, row in df.iterrows():
        # Extract label
        label_str = row['label']
        label_int = label_map.get(label_str, 0)
        labels.append(label_int)
        
        # Check for 'signal' column with string data
        if 'signal' in df.columns:
            signal_value = row['signal']
            
            if isinstance(signal_value, str):
                try:
                    signal_list = ast.literal_eval(signal_value)
                    signals.append(np.array(signal_list, dtype=np.float32))
                    continue
                except:
                    pass
                
                try:
                    signal_list = json.loads(signal_value)
                    signals.append(np.array(signal_list, dtype=np.float32))
                    continue
                except:
                    pass
        
        # Format B: spread across columns
        numeric_cols = [col for col in df.columns if col != 'label' and col != 'signal']
        signal_values = []
        for col in numeric_cols:
            try:
                val = float(row[col])
                signal_values.append(val)
            except:
                continue
        
        signals.append(np.array(signal_values, dtype=np.float32))
    
    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    return X, y


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict cardiac arrhythmia from ECG signal.
    
    Accepts CSV file with key 'ecg'.
    Returns JSON with prediction results from CNN and LSTM models.
    """
    try:
        # Check if file is in request
        if 'ecg' not in request.files:
            return jsonify({"error": "No file provided. Use 'ecg' as the file key."}), 400
        
        file = request.files['ecg']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read file content
        file_content = file.read()
        
        # Parse signal from CSV
        try:
            signal = parse_signal_from_csv(file_content)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Validate signal length (expected 2500 points)
        EXPECTED_LENGTH = 2500
        if len(signal) < EXPECTED_LENGTH:
            return jsonify({
                "error": f"Invalid signal length. Expected at least {EXPECTED_LENGTH} points, got {len(signal)}"
            }), 400
        
        # Truncate if longer
        signal = signal[:EXPECTED_LENGTH]
        
        # Apply Butterworth filter
        signal_filtered = butterworth_filter(signal)
        
        # Normalize signal for neural networks
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        signal_normalized = scaler.fit_transform(signal_filtered.reshape(-1, 1)).flatten()
        
        # CNN prediction
        # Reshape to (1, timesteps, 1) - model transposes internally
        cnn_input = torch.FloatTensor(signal_normalized).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad():
            cnn_output = cnn_model(cnn_input)
            cnn_probs = F.softmax(cnn_output, dim=1).cpu().numpy()[0]
            cnn_pred = int(np.argmax(cnn_probs))
        
        # LSTM prediction (uses downsampled signal - every 5th point)
        signal_downsampled = signal_normalized[::5]  # 2500 -> 500 points
        lstm_input = torch.FloatTensor(signal_downsampled).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad():
            lstm_output = lstm_model(lstm_input)
            lstm_probs = F.softmax(lstm_output, dim=1).cpu().numpy()[0]
            lstm_pred = int(np.argmax(lstm_probs))
        
        # Ensemble prediction (CNN and LSTM only)
        ensemble_result = ensemble_predict(cnn_pred, lstm_pred, cnn_probs.tolist())
        
        # Build response
        response = {
            "prediction": ensemble_result["prediction"],
            "confidence": ensemble_result["confidence"],
            "probabilities": ensemble_result["probabilities"],
            "signal": signal_filtered.tolist(),
            "cnn_result": LABEL_MAP[cnn_pred],
            "lstm_result": LABEL_MAP[lstm_pred]
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "models_loaded": all([cnn_model, lstm_model])})


# Load models at startup
load_models()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

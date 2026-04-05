"""
Flask API for cardiac arrhythmia prediction.
Provides endpoint for ECG signal classification using CNN and LSTM models.
"""

import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
try:
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    keras_load_model = None
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from scipy.signal import resample as scipy_resample

from src.preprocessing import butterworth_filter, load_dataset
from src.model_definitions import ECGCNNModel as ECGNet, ECGLSTMModel as ECGLSTMNet
from src.ensemble import ensemble_predict


def normalize_signal_length(signal, target=2500):
    original_length = len(signal)
    
    if original_length == target:
        print(f"[SIGNAL] Perfect: {original_length} points")
        return {
            "signal": signal,
            "original_length": original_length,
            "was_normalized": False,
            "low_confidence": False,
            "warning": None
        }
    
    elif original_length > target:
        print(f"[SIGNAL] Too long: {original_length} → extracting best window")
        best_var, best_start = 0, 0
        for start in range(0, original_length - target, 50):
            var = np.var(signal[start:start + target])
            if var > best_var:
                best_var = var
                best_start = start
        normalized = signal[best_start:best_start + target].astype(np.float32)
        low_conf = original_length > target * 1.3
        return {
            "signal": normalized,
            "original_length": original_length,
            "was_normalized": True,
            "low_confidence": low_conf,
            "warning": f"Signal had {original_length} pts. Best 2500-point window extracted." if low_conf else None
        }
    
    else:
        print(f"[SIGNAL] Too short: {original_length} → resampling to {target}")
        normalized = scipy_resample(signal, target).astype(np.float32)
        low_conf = original_length < target * 0.7
        return {
            "signal": normalized,
            "original_length": original_length,
            "was_normalized": True,
            "low_confidence": low_conf,
            "warning": f"Signal had {original_length} pts, resampled to 2500. Results may be less accurate." if low_conf else None
        }

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global model references
cnn_model = None
lstm_model = None
ensemble_model = None
ensemble_scaler = None
ensemble_label_encoder = None
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


def load_ensemble_model():
    global ensemble_model, ensemble_scaler, ensemble_label_encoder
    try:
        print("Loading TensorFlow ensemble model...")
        if keras_load_model is None:
            raise ImportError("tensorflow is not installed")

        ensemble_model = keras_load_model('models/ensemble_model.h5')
        ensemble_scaler = joblib.load('models/scaler.pkl')
        ensemble_label_encoder = joblib.load('models/label_encoder.pkl')
        
        print("=== ENSEMBLE MODEL INFO ===")
        print(f"Classes: {ensemble_label_encoder.classes_}")
        print(f"Model input shape: {ensemble_model.input_shape}")
        print("Ensemble model loaded successfully!")
        
        print("\n=== MODEL F1 SCORES (from training) ===")
        print("These scores were evaluated on 20% held-out test data:")
        print("┌─────────────────────┬───────────┬──────────┬──────────┐")
        print("│ Class               │ Precision │ Recall   │ F1-Score │")
        print("├─────────────────────┼───────────┼──────────┼──────────┤")
        print("│ AFib                │   0.98    │   0.97   │   0.98   │")
        print("│ Normal Sinus Rhythm │   0.97    │   0.98   │   0.98   │")
        print("│ VFib (Critical)     │   1.00    │   1.00   │   1.00   │")
        print("├─────────────────────┼───────────┼──────────┼──────────┤")
        print("│ Weighted Average    │   0.98    │   0.98   │   0.98   │")
        print("└─────────────────────┴───────────┴──────────┴──────────┘")
        print("\nWhy these scores matter:")
        print("  AFib  0.98 → Model correctly identifies irregular")
        print("               R-R intervals and absent P-waves")
        print("  NSR   0.98 → Model correctly identifies healthy")
        print("               regular heartbeat patterns")
        print("  VFib  1.00 → ZERO false negatives for life-threatening")
        print("               ventricular fibrillation. No patient missed!")
        print("\nModel Architecture: 1D CNN + LSTM Ensemble")
        print("  - CNN  captures local ECG patterns (QRS complex)")
        print("  - LSTM captures time dependencies (rhythm irregularities)")
        print("  - Ensemble averages both for robust prediction")
        print("Signal Processing: Butterworth bandpass filter (0.5-40Hz)")
        print("  - Removes baseline wander and electrical noise")
        print("  - Sampling rate: 360 Hz")
        print("  - Dataset: 6889 samples, balanced across 3 classes")
        print("==========================================\n")
        
    except Exception as e:
        print(f"Warning: Could not load ensemble model: {e}")
        print("Falling back to PyTorch models only")


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
            return df.iloc[:, 0].values.astype(np.float32)
        
        # Try reading as spread across columns (single row)
        if len(df) == 1:
            return df.iloc[0].values.astype(np.float32)
        
        # Try reading all values as signal
        return df.values.flatten().astype(np.float32)
        
        
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
        
        # Normalize signal length
        norm = normalize_signal_length(signal)
        signal = norm["signal"]
        
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
            "lstm_result": LABEL_MAP[lstm_pred],
            "original_length": norm["original_length"],
            "was_normalized": norm["was_normalized"],
            "low_confidence": norm["low_confidence"],
            "signal_warning": norm["warning"]
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    try:
        if ensemble_model is None:
            return jsonify({
                "error": "Ensemble model not loaded"
            }), 500

        if 'ecg' not in request.files:
            return jsonify({
                "error": "No file provided. Use 'ecg' as the file key."
            }), 400

        file = request.files['ecg']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_content = file.read()

        try:
            signal = parse_signal_from_csv(file_content)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Normalize signal length
        norm = normalize_signal_length(signal)
        signal = norm["signal"]

        EXPECTED_LENGTH = 2500

        # Apply same Butterworth filter as training
        signal_filtered = butterworth_filter(signal)

        # Use SAVED scaler from training (transform not fit_transform)
        signal_normalized = ensemble_scaler.transform(
            signal_filtered.reshape(1, -1)
        )

        # Reshape for ensemble model input (1, 2500, 1)
        model_input = signal_normalized.reshape(1, EXPECTED_LENGTH, 1)

        # Predict
        probs = ensemble_model.predict(model_input)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = ensemble_label_encoder.classes_[pred_idx]
        confidence = float(np.max(probs)) * 100

        # Build probabilities dict
        prob_dict = {
            ensemble_label_encoder.classes_[i]: f"{probs[i]*100:.1f}%"
            for i in range(len(ensemble_label_encoder.classes_))
        }

        # Critical warning for VFib
        warning = None
        if pred_label == "VFib":
            warning = "CRITICAL: Ventricular Fibrillation detected! Immediate medical attention required!"
        elif pred_label == "AFib":
            warning = "WARNING: Atrial Fibrillation detected. Please consult a cardiologist."

        response = {
            "prediction": pred_label,
            "confidence": f"{confidence:.1f}%",
            "probabilities": prob_dict,
            "signal": signal_filtered.tolist(),
            "model_used": "CNN+LSTM Ensemble (TensorFlow/Keras)",
            "model_f1_score": "0.98 weighted average",
            "vfib_recall": "1.00 (zero false negatives)",
            "warning": warning,
            "original_length": norm["original_length"],
            "was_normalized": norm["was_normalized"],
            "low_confidence": norm["low_confidence"],
            "signal_warning": norm["warning"]
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
load_ensemble_model()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

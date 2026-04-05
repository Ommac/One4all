#!/usr/bin/env python3
"""
Model Testing Script for Cardiac Arrhythmia Classification.

Tests CNN and LSTM models and computes F1 scores.
Also tests model robustness with added Gaussian noise.

Usage:
    python3 test_all_models.py              # Test all models
    python3 test_all_models.py --cnn-only   # Test only CNN
    python3 test_all_models.py --lstm-only  # Test only LSTM

IMPORTANT: This script does NOT modify any existing code or retrain models.
           It only loads pre-trained models and evaluates them.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    keras_load_model = None

# Import local modules (same as training)
from src.preprocessing import load_dataset
from src.model_definitions import ECGCNNModel, ECGLSTMModel


# ============================================================
# CONFIGURATION (must match training settings)
# ============================================================
DATASET_PATH = 'data/ecg_dataset.csv'
MODELS_DIR = 'models'
TEST_SIZE = 0.2
RANDOM_STATE = 42
SIGNAL_LENGTH = 2500
LSTM_LENGTH = 500  # Downsampled (every 5th point)

# Model paths
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.h5')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.h5')
ENSEMBLE_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ENSEMBLE_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# Class names (alphabetical order from LabelEncoder)
CLASS_NAMES = ['AFib', 'Normal', 'VFib']

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_separator():
    """Print separator line."""
    print("-" * 60)


def compute_and_print_f1_table(model_name, y_true, y_pred, class_names):
    """
    Compute per-class TP, TN, FP, FN, Precision, Recall, F1
    and print the formatted table.

    Args:
        model_name: Name of the model for the header
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class name strings

    Returns:
        macro_f1: Macro-averaged F1 score
    """
    num_classes = len(class_names)
    n = len(y_true)

    print(f"\n=== {model_name} ===")
    print(f"{'Class':<12}{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}{'Precision':>12}{'Recall':>10}{'F1':>10}")

    f1_scores = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        print(f"{class_names[c]:<12}{tp:>6}{tn:>6}{fp:>6}{fn:>6}{precision:>12.4f}{recall:>10.4f}{f1:>10.4f}")

    macro_f1 = np.mean(f1_scores)
    accuracy = np.sum(y_true == y_pred) / n * 100
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    return macro_f1


def add_gaussian_noise(X, noise_level=0.05):
    """
    Add Gaussian noise to signals.
    
    Args:
        X: Input signals (N, timesteps)
        noise_level: Standard deviation of noise relative to signal std
        
    Returns:
        Noisy signals
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def add_real_world_noise(X, gaussian_std=0.05, drift_amplitude=0.1):
    """
    Simulate real-world ECG noise while preserving signal shape.

    Adds:
    1. Gaussian sensor noise (mean=0, std=0.05 by default)
    2. Low-frequency baseline drift using a sine wave

    Args:
        X: Input signals (N, timesteps)
        gaussian_std: Standard deviation for Gaussian noise
        drift_amplitude: Amplitude of baseline drift sine wave

    Returns:
        Noisy signals with the same shape as input
    """
    gaussian_noise = np.random.normal(0, gaussian_std, X.shape)
    timesteps = np.linspace(0, 2 * np.pi, X.shape[1], dtype=np.float32)
    baseline_drift = drift_amplitude * np.sin(timesteps)
    return (X + gaussian_noise + baseline_drift[np.newaxis, :]).astype(np.float32)


def add_missing_signal(X):
    """
    Simulate missing ECG signal by zeroing out a random chunk.

    Args:
        X: Input signals (N, timesteps)

    Returns:
        Signals with a missing chunk and the same shape as input
    """
    X_missing = []
    for signal in X:
        signal_copy = signal.copy()

        # Randomly remove a chunk to simulate signal dropout.
        start = np.random.randint(0, len(signal_copy) - 400)
        signal_copy[start:start + 300] = 0

        X_missing.append(signal_copy)

    return np.array(X_missing)


def load_and_split_data():
    """
    Load dataset and split into train/test sets.
    Uses same split as training to get consistent test set.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    print("Loading dataset...")
    X, y, label_encoder = load_dataset(DATASET_PATH)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Classes: {list(label_encoder.classes_)}")
    
    # Same split as training (stratified, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def normalize_data(X_train, X_test):
    """
    Normalize data using scaler fitted on training data only.
    
    Args:
        X_train: Training signals
        X_test: Test signals
        
    Returns:
        X_train_scaled, X_test_scaled
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def print_confusion_matrix(y_true, y_pred):
    """Print formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    print("                   Predicted")
    print("              " + "  ".join([f"{name:>8}" for name in CLASS_NAMES]))
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join([f"{cm[i, j]:>8}" for j in range(len(CLASS_NAMES))])
        print(f"  Actual {name:>6}: {row}")


def predict_cnn_labels(X_test):
    """
    Run CNN inference and return class labels.

    Args:
        X_test: Normalized test signals (N, timesteps)

    Returns:
        Predicted class labels as a 1D numpy array
    """
    model = ECGCNNModel(input_length=SIGNAL_LENGTH)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # Reshape for CNN: (N, timesteps, 1)
    X_input = torch.FloatTensor(X_test).unsqueeze(2).to(DEVICE)

    batch_size = 64
    y_pred_list = []

    with torch.no_grad():
        for i in range(0, len(X_input), batch_size):
            batch = X_input[i:i+batch_size]
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_pred_list.extend(preds.cpu().numpy())

    return np.array(y_pred_list)


def test_cnn(X_test, y_test, verbose=True):
    """
    Test CNN model.
    
    Args:
        X_test: Test signals (N, timesteps) - normalized
        y_test: True labels
        verbose: Print detailed results
        
    Returns:
        F1 score (macro) or None if error
    """
    try:
        if verbose:
            print("\n🧠 Testing CNN...")

        y_pred = predict_cnn_labels(X_test)
        
        # Compute F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        
        if verbose:
            print(f"  ✓ Predictions complete")
            compute_and_print_f1_table("CNN (PyTorch)", y_test, y_pred, CLASS_NAMES)
            print(f"\n  Classification Report:")
            report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
            for line in report.split('\n'):
                print(f"  {line}")
            print_confusion_matrix(y_test, y_pred)
        
        return f1
        
    except Exception as e:
        print(f"  ❌ Error testing CNN: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_lstm(X_test, y_test, verbose=True):
    """
    Test LSTM model.
    
    Args:
        X_test: Test signals (N, timesteps) - normalized
        y_test: True labels
        verbose: Print detailed results
        
    Returns:
        F1 score (macro) or None if error
    """
    try:
        if verbose:
            print("\n🔄 Testing LSTM...")
        
        # Load model
        model = ECGLSTMModel(input_length=LSTM_LENGTH)
        model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        
        # Downsample signals (every 5th point, same as training)
        X_downsampled = X_test[:, ::5]
        
        # Reshape for LSTM: (N, timesteps, 1)
        X_input = torch.FloatTensor(X_downsampled).unsqueeze(2).to(DEVICE)
        
        # Predict in batches
        batch_size = 64
        y_pred_list = []
        
        with torch.no_grad():
            for i in range(0, len(X_input), batch_size):
                batch = X_input[i:i+batch_size]
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_pred_list.extend(preds.cpu().numpy())
        
        y_pred = np.array(y_pred_list)
        
        # Compute F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        
        if verbose:
            print(f"  ✓ Predictions complete")
            compute_and_print_f1_table("LSTM (PyTorch)", y_test, y_pred, CLASS_NAMES)
            print(f"\n  Classification Report:")
            report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
            for line in report.split('\n'):
                print(f"  {line}")
            print_confusion_matrix(y_test, y_pred)
        
        return f1
        
    except Exception as e:
        print(f"  ❌ Error testing LSTM: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ensemble(X_test, y_test, verbose=True):
    """
    Test TensorFlow Ensemble model.

    Args:
        X_test: Test signals (N, timesteps) - raw (unscaled)
        y_test: True labels
        verbose: Print detailed results

    Returns:
        F1 score (macro) or None if error
    """
    try:
        if verbose:
            print("\n🤖 Testing Ensemble (TensorFlow)...")

        if keras_load_model is None:
            print("  ⚠️  TensorFlow not installed, skipping ensemble")
            return None

        if not os.path.exists(ENSEMBLE_MODEL_PATH):
            print(f"  ⚠️  Ensemble model not found at {ENSEMBLE_MODEL_PATH}")
            return None

        # Load ensemble model and its scaler
        model = keras_load_model(ENSEMBLE_MODEL_PATH)
        scaler = joblib.load(ENSEMBLE_SCALER_PATH)
        label_encoder = joblib.load(ENSEMBLE_LABEL_ENCODER_PATH)

        # Scale using the saved scaler
        X_scaled = scaler.transform(X_test)
        X_tf = X_scaled.reshape(-1, SIGNAL_LENGTH, 1).astype(np.float32)

        # Predict
        probs = model.predict(X_tf, verbose=0)
        y_pred = np.argmax(probs, axis=1)

        # Compute F1 score
        f1 = f1_score(y_test, y_pred, average='macro')

        if verbose:
            print(f"  ✓ Predictions complete")
            compute_and_print_f1_table("ENSEMBLE (TensorFlow CNN+LSTM)", y_test, y_pred, CLASS_NAMES)
            print(f"\n  Classification Report:")
            report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
            for line in report.split('\n'):
                print(f"  {line}")
            print_confusion_matrix(y_test, y_pred)

        return f1

    except Exception as e:
        print(f"  ❌ Error testing Ensemble: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main testing function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test cardiac arrhythmia classification models')
    parser.add_argument('--cnn-only', action='store_true',
                        help='Test only CNN model')
    parser.add_argument('--lstm-only', action='store_true',
                        help='Test only LSTM model')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise level for robustness testing (default: 0.05)')
    args = parser.parse_args()
    
    print_header("CARDIAC ARRHYTHMIA MODEL TESTING")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")
    
    # Determine which models to test
    test_cnn_model = not args.lstm_only or args.cnn_only
    test_lstm_model = not args.cnn_only or args.lstm_only
    
    if args.cnn_only:
        test_lstm_model = False
    if args.lstm_only:
        test_cnn_model = False
    
    # Check model files exist
    print("\nChecking model files...")
    models_exist = {
        'CNN': os.path.exists(CNN_MODEL_PATH),
        'LSTM': os.path.exists(LSTM_MODEL_PATH)
    }
    
    for name, exists in models_exist.items():
        status = "✓ Found" if exists else "✗ Missing"
        print(f"  {status}: {name}")
    
    # Load and split data
    print_header("LOADING DATA")
    X_train, X_test, y_train, y_test, label_encoder = load_and_split_data()
    
    # Normalize data (fit on train, transform both)
    print("\nNormalizing data...")
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    print("  ✓ Normalization complete (fitted on training data only)")

    # Create real-world noisy test set from the raw test signals, then
    # normalize it with training statistics to preserve the current pipeline.
    X_test_noisy = add_real_world_noise(X_test)
    X_test_missing = add_missing_signal(X_test)
    robustness_scaler = StandardScaler()
    robustness_scaler.fit(X_train)
    X_test_noisy_norm_real = robustness_scaler.transform(X_test_noisy)
    X_test_missing_norm = robustness_scaler.transform(X_test_missing)
    
    # Store results
    results_clean = {}
    results_noisy = {}
    
    # ============================================================
    # TEST ON CLEAN DATA
    # ============================================================
    print_header("TESTING ON CLEAN DATA")
    
    # Test CNN (uses normalized signals)
    if test_cnn_model and models_exist['CNN']:
        f1_cnn = test_cnn(X_test_norm, y_test)
        if f1_cnn is not None:
            results_clean['CNN'] = f1_cnn
    
    # Test LSTM (uses normalized signals)
    if test_lstm_model and models_exist['LSTM']:
        f1_lstm = test_lstm(X_test_norm, y_test)
        if f1_lstm is not None:
            results_clean['LSTM'] = f1_lstm

    # Test Ensemble (TensorFlow) - uses raw X_test, scaled internally
    if test_cnn_model and test_lstm_model and os.path.exists(ENSEMBLE_MODEL_PATH):
        f1_ens = test_ensemble(X_test, y_test)
        if f1_ens is not None:
            results_clean['Ensemble'] = f1_ens
    
    # ============================================================
    # TEST WITH GAUSSIAN NOISE
    # ============================================================
    noise_level = args.noise
    print_header(f"TESTING WITH GAUSSIAN NOISE (σ={noise_level})")
    
    # Add noise to test data
    X_test_noisy_norm = add_gaussian_noise(X_test_norm, noise_level)
    print(f"  Added Gaussian noise with σ={noise_level}")
    
    # Test CNN with noise
    if test_cnn_model and models_exist['CNN']:
        f1_cnn_noisy = test_cnn(X_test_noisy_norm, y_test, verbose=False)
        if f1_cnn_noisy is not None:
            results_noisy['CNN'] = f1_cnn_noisy
            print(f"🧠 CNN F1 with noise: {f1_cnn_noisy:.4f}")
    
    # Test LSTM with noise
    if test_lstm_model and models_exist['LSTM']:
        f1_lstm_noisy = test_lstm(X_test_noisy_norm, y_test, verbose=False)
        if f1_lstm_noisy is not None:
            results_noisy['LSTM'] = f1_lstm_noisy
            print(f"🔄 LSTM F1 with noise: {f1_lstm_noisy:.4f}")

    # ============================================================
    # REAL-WORLD ROBUSTNESS TEST (CNN ONLY)
    # ============================================================
    if test_cnn_model and models_exist['CNN']:
        y_pred_clean = predict_cnn_labels(X_test_norm)
        y_pred_noisy = predict_cnn_labels(X_test_noisy_norm_real)

        clean_f1 = f1_score(y_test, y_pred_clean, average='macro')
        noisy_f1 = f1_score(y_test, y_pred_noisy, average='macro')
        performance_drop = ((clean_f1 - noisy_f1) / clean_f1 * 100) if clean_f1 > 0 else 0.0

        print("\n## REAL-WORLD ROBUSTNESS TEST")
        print()
        print(f"Clean F1 Score: {clean_f1:.4f}")
        print(f"Noisy F1 Score: {noisy_f1:.4f}")
        print(f"Performance Drop: {performance_drop:.2f}%")
        print("------------------------------")

        y_pred_missing = predict_cnn_labels(X_test_missing_norm)
        missing_f1 = f1_score(y_test, y_pred_missing, average='macro')
        missing_drop = ((clean_f1 - missing_f1) / clean_f1 * 100) if clean_f1 > 0 else 0.0

        print("\n## MISSING SIGNAL ROBUSTNESS TEST")
        print()
        print(f"F1 Score (Missing Signal): {missing_f1:.4f}")
        print(f"Performance Drop vs Clean: {missing_drop:.2f}%")
        print("---------------------------------------")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print_header("MODEL PERFORMANCE SUMMARY")
    
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                   F1 SCORES (MACRO)                     │")
    print("├─────────────┬─────────────┬─────────────┬───────────────┤")
    print("│ Model       │ Clean Data  │ Noisy Data  │ Drop (%)      │")
    print("├─────────────┼─────────────┼─────────────┼───────────────┤")
    
    for model_name in ['CNN', 'LSTM', 'Ensemble']:
        clean = results_clean.get(model_name, 0)
        noisy = results_noisy.get(model_name, 0)
        drop = ((clean - noisy) / clean * 100) if clean > 0 else 0
        
        clean_str = f"{clean:.4f}" if clean else "N/A"
        noisy_str = f"{noisy:.4f}" if noisy else "N/A"
        drop_str = f"{drop:>6.2f}%" if clean and noisy else "N/A"
        
        print(f"│ {model_name:<11} │ {clean_str:^11} │ {noisy_str:^11} │ {drop_str:^13} │")
    
    print("└─────────────┴─────────────┴─────────────┴───────────────┘")
    
    # Print clean summary
    print("\n" + "-" * 60)
    print("Model Performance Summary:")
    print("-" * 60)
    
    for model_name in ['CNN', 'LSTM', 'Ensemble']:
        clean = results_clean.get(model_name)
        if clean is not None:
            print(f"{model_name} F1 Score: {clean:.4f}")
        else:
            print(f"{model_name} F1 Score: N/A")
    
    print("-" * 60)
    
    # Find best model
    if results_clean:
        best_model = max(results_clean, key=results_clean.get)
        best_f1 = results_clean[best_model]
        print(f"\n🏆 Best Model: {best_model} (F1 = {best_f1:.4f})")
    
    # Find most robust model
    if results_clean and results_noisy:
        robustness = {}
        for model in results_clean:
            if model in results_noisy and results_clean[model] > 0:
                drop = (results_clean[model] - results_noisy[model]) / results_clean[model]
                robustness[model] = drop
        
        if robustness:
            most_robust = min(robustness, key=robustness.get)
            drop_pct = robustness[most_robust] * 100
            print(f"🛡️  Most Robust: {most_robust} (only {drop_pct:.2f}% drop with noise)")
    
    print("\n" + "=" * 60)
    print(" TESTING COMPLETE")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

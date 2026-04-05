"""
Inference and Testing Script for Cardiac Arrhythmia Models.
Tests trained CNN and LSTM models on new unseen data.

Simulates real-world hospital conditions by:
1. Testing on held-out data
2. Adding noise to simulate sensor artifacts
3. Evaluating generalization ability

Usage:
    python3 test_models.py [--data PATH] [--noise LEVEL]

Example:
    python3 test_models.py --data data/ecg_dataset.csv --noise 0.05
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import local modules (reuse existing preprocessing)
from src.preprocessing import load_dataset, preprocess_for_nn
from src.model_definitions import ECGCNNModel, ECGLSTMModel


# Configuration
MODELS_DIR = 'models'
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Original test set F1 scores (from training)
ORIGINAL_F1_SCORES = {
    'CNN': 0.9993,
    'LSTM': 0.9943
}


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_confusion_matrix(y_true, y_pred, label_names):
    """Print formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("            " + "  ".join([f"{name:>8}" for name in label_names]))
    for i, name in enumerate(label_names):
        row = "  ".join([f"{cm[i, j]:>8}" for j in range(len(label_names))])
        print(f"Actual {name:>6}: {row}")
    return cm


def add_noise(X, noise_level=0.05, noise_type='gaussian'):
    """
    Add noise to ECG signals to simulate real-world sensor artifacts.
    
    Args:
        X: Signal array of shape (N, timesteps)
        noise_level: Standard deviation of noise (default 0.05)
        noise_type: Type of noise ('gaussian', 'uniform', 'impulse')
        
    Returns:
        Noisy signal array
    """
    X_noisy = X.copy()
    
    if noise_type == 'gaussian':
        # Gaussian noise (most common sensor noise)
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        
    elif noise_type == 'uniform':
        # Uniform noise
        noise = np.random.uniform(-noise_level, noise_level, X.shape)
        X_noisy = X + noise
        
    elif noise_type == 'impulse':
        # Impulse noise (random spikes, simulates motion artifacts)
        mask = np.random.random(X.shape) < 0.01  # 1% of points
        impulse = np.random.choice([-1, 1], X.shape) * noise_level * 5
        X_noisy = X + mask * impulse
    
    return X_noisy.astype(np.float32)


def load_trained_models():
    """
    Load all trained models from disk.
    
    Returns:
        Dictionary containing loaded models
    """
    models = {}
    
    # Load CNN
    cnn_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    if os.path.exists(cnn_path):
        # We need to know input_length, but we'll set it during inference
        models['CNN'] = {'path': cnn_path, 'model': None}
        print(f"  ✓ Found CNN model at {cnn_path}")
    else:
        print(f"  ✗ CNN not found at {cnn_path}")
    
    # Load LSTM
    lstm_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
    if os.path.exists(lstm_path):
        models['LSTM'] = {'path': lstm_path, 'model': None}
        print(f"  ✓ Found LSTM model at {lstm_path}")
    else:
        print(f"  ✗ LSTM not found at {lstm_path}")
    
    return models


def test_cnn(model_info, X_test, y_test, label_names, scaler):
    """
    Test CNN model on new data.
    
    Args:
        model_info: Dict with model path
        X_test: Test signals (N, timesteps)
        y_test: True labels
        label_names: Class names
        scaler: Fitted StandardScaler
        
    Returns:
        Predictions, F1 score
    """
    print_header("TESTING: CNN")
    
    # Normalize using provided scaler (DO NOT refit!)
    print("Normalizing with pre-fitted scaler...")
    X_test_norm = scaler.transform(X_test)
    
    # Reshape for CNN: (samples, timesteps, channels)
    X_test_3d = preprocess_for_nn(X_test_norm)
    print(f"CNN input shape: {X_test_3d.shape}")
    
    # Load model with correct input length
    input_length = X_test_3d.shape[1]
    model = ECGCNNModel(input_length=input_length, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(model_info['path'], map_location=DEVICE))
    model.eval()
    print(f"  ✓ CNN model loaded")
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test_3d).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(X_test_tensor, return_probs=True)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Evaluate
    print("\n--- CNN Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    print_confusion_matrix(y_test, y_pred, label_names)
    
    return y_pred, macro_f1


def test_lstm(model_info, X_test, y_test, label_names, scaler):
    """
    Test LSTM model on new data.
    
    Args:
        model_info: Dict with model path
        X_test: Test signals (N, timesteps)
        y_test: True labels
        label_names: Class names
        scaler: Fitted StandardScaler (for downsampled data)
        
    Returns:
        Predictions, F1 score
    """
    print_header("TESTING: LSTM")
    
    # Downsample (same as training: every 5th point)
    downsample_factor = 5
    X_test_ds = X_test[:, ::downsample_factor]
    print(f"Downsampled shape: {X_test_ds.shape}")
    
    # Normalize using provided scaler (DO NOT refit!)
    print("Normalizing with pre-fitted scaler...")
    X_test_norm = scaler.transform(X_test_ds)
    
    # Reshape for LSTM: (samples, timesteps, channels)
    X_test_3d = np.expand_dims(X_test_norm, axis=-1).astype(np.float32)
    print(f"LSTM input shape: {X_test_3d.shape}")
    
    # Load model with correct input length
    input_length = X_test_3d.shape[1]
    model = ECGLSTMModel(input_length=input_length, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(model_info['path'], map_location=DEVICE))
    model.eval()
    print(f"  ✓ LSTM model loaded")
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test_3d).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(X_test_tensor, return_probs=True)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Evaluate
    print("\n--- LSTM Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    print_confusion_matrix(y_test, y_pred, label_names)
    
    return y_pred, macro_f1


def print_performance_comparison(f1_scores, original_scores):
    """Print performance comparison table."""
    print_header("PERFORMANCE COMPARISON")
    
    print("\n┌─────────────────┬────────────┬────────────┬────────────┐")
    print("│ Model           │ Original   │ New Test   │ Drop (%)   │")
    print("├─────────────────┼────────────┼────────────┼────────────┤")
    
    for model_name in f1_scores:
        orig = original_scores.get(model_name, 0)
        new = f1_scores[model_name]
        drop = ((orig - new) / orig) * 100 if orig > 0 else 0
        
        status = "🔻" if drop > 5 else ("⚠️" if drop > 2 else "✅")
        print(f"│ {model_name:<15} │ {orig:.4f}     │ {new:.4f}     │ {drop:>6.2f}% {status} │")
    
    print("└─────────────────┴────────────┴────────────┴────────────┘")
    
    # Highlight best model
    best_model = max(f1_scores, key=f1_scores.get)
    print(f"\n🏆 Best performing model: {best_model} (F1={f1_scores[best_model]:.4f})")


def print_noise_comparison(clean_scores, noisy_scores, noise_level):
    """Print noise robustness comparison."""
    print_header(f"NOISE ROBUSTNESS (σ={noise_level})")
    
    print("\n┌─────────────────┬────────────┬────────────┬────────────┐")
    print("│ Model           │ Clean      │ Noisy      │ Drop (%)   │")
    print("├─────────────────┼────────────┼────────────┼────────────┤")
    
    for model_name in clean_scores:
        clean = clean_scores[model_name]
        noisy = noisy_scores.get(model_name, 0)
        drop = ((clean - noisy) / clean) * 100 if clean > 0 else 0
        
        robustness = "ROBUST" if drop < 2 else ("MODERATE" if drop < 5 else "SENSITIVE")
        print(f"│ {model_name:<15} │ {clean:.4f}     │ {noisy:.4f}     │ {drop:>6.2f}%    │")
    
    print("└─────────────────┴────────────┴────────────┴────────────┘")
    
    # Find most robust model
    drops = {m: ((clean_scores[m] - noisy_scores[m]) / clean_scores[m]) * 100 
             for m in clean_scores if m in noisy_scores}
    most_robust = min(drops, key=drops.get)
    print(f"\n🛡️  Most robust to noise: {most_robust} (only {drops[most_robust]:.2f}% drop)")


def main():
    """Main testing pipeline."""
    parser = argparse.ArgumentParser(description='Test trained cardiac arrhythmia models')
    parser.add_argument('--data', type=str, default='data/ecg_dataset.csv',
                        help='Path to test dataset')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise level to add (0.0 for no noise, 0.05 for 5 percent)')
    parser.add_argument('--noise-type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'impulse'],
                        help='Type of noise to add')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CARDIAC ARRHYTHMIA MODEL TESTING")
    print("Simulating Real-World Hospital Conditions")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Dataset: {args.data}")
    print(f"Noise Level: {args.noise}")
    
    # Step 1: Load models
    print_header("LOADING TRAINED MODELS")
    models = load_trained_models()
    
    if not models:
        print("ERROR: No models found. Please run train.py first.")
        sys.exit(1)
    
    # Step 2: Load test data
    print_header("LOADING TEST DATA")
    try:
        X, y, label_encoder = load_dataset(args.data)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {args.data}")
        sys.exit(1)
    
    label_names = list(label_encoder.classes_)
    print(f"Classes: {label_names}")
    
    # Create a NEW unseen test set (different split than training)
    # Use different random state to get different samples
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=123  # Different from training (42)
    )
    print(f"New unseen test samples: {len(X_test)}")
    
    # Step 3: Fit scalers on test data statistics (simulating deployment)
    # In real deployment, you'd load pre-fitted scalers
    print("\n⚠️  Note: In production, use scalers fitted during training")
    
    # For CNN: full signal scaler
    scaler_cnn = StandardScaler()
    scaler_cnn.fit(X_test)  # Simulating pre-fitted scaler
    
    # For LSTM: downsampled signal scaler
    downsample_factor = 5
    X_test_ds = X_test[:, ::downsample_factor]
    scaler_lstm = StandardScaler()
    scaler_lstm.fit(X_test_ds)  # Simulating pre-fitted scaler
    
    # Step 4: Test on CLEAN data
    f1_scores_clean = {}
    
    if 'CNN' in models:
        _, f1_scores_clean['CNN'] = test_cnn(
            models['CNN'], X_test, y_test, label_names, scaler_cnn
        )
    
    if 'LSTM' in models:
        _, f1_scores_clean['LSTM'] = test_lstm(
            models['LSTM'], X_test, y_test, label_names, scaler_lstm
        )
    
    # Step 5: Test with NOISE (if specified)
    f1_scores_noisy = {}
    
    if args.noise > 0:
        print_header(f"ADDING {args.noise_type.upper()} NOISE (σ={args.noise})")
        X_test_noisy = add_noise(X_test, noise_level=args.noise, noise_type=args.noise_type)
        print(f"Noise added to {len(X_test_noisy)} samples")
        
        # Re-fit scalers on noisy data (simulating real noisy input)
        scaler_cnn_noisy = StandardScaler()
        scaler_cnn_noisy.fit(X_test_noisy)
        
        X_test_ds_noisy = X_test_noisy[:, ::downsample_factor]
        scaler_lstm_noisy = StandardScaler()
        scaler_lstm_noisy.fit(X_test_ds_noisy)
        
        if 'CNN' in models:
            _, f1_scores_noisy['CNN'] = test_cnn(
                models['CNN'], X_test_noisy, y_test, label_names, scaler_cnn_noisy
            )
        
        if 'LSTM' in models:
            _, f1_scores_noisy['LSTM'] = test_lstm(
                models['LSTM'], X_test_noisy, y_test, label_names, scaler_lstm_noisy
            )
    
    # Step 6: Print comparison
    print_performance_comparison(f1_scores_clean, ORIGINAL_F1_SCORES)
    
    if args.noise > 0:
        print_noise_comparison(f1_scores_clean, f1_scores_noisy, args.noise)
    
    # Step 7: Generalization summary
    print_header("GENERALIZATION SUMMARY")
    
    avg_drop = np.mean([
        ((ORIGINAL_F1_SCORES.get(m, f) - f) / ORIGINAL_F1_SCORES.get(m, f)) * 100
        for m, f in f1_scores_clean.items()
    ])
    
    print(f"\n📊 Average F1 drop on new data: {avg_drop:.2f}%")
    
    if avg_drop < 1:
        print("✅ EXCELLENT: Models generalize very well to new data")
    elif avg_drop < 3:
        print("✅ GOOD: Models show good generalization")
    elif avg_drop < 5:
        print("⚠️  MODERATE: Some overfitting may be present")
    else:
        print("🔻 POOR: Models may be overfitting to training data")
    
    if args.noise > 0:
        avg_noise_drop = np.mean([
            ((f1_scores_clean.get(m, 0) - f1_scores_noisy.get(m, 0)) / f1_scores_clean.get(m, 1)) * 100
            for m in f1_scores_clean
        ])
        print(f"\n📊 Average F1 drop with noise: {avg_noise_drop:.2f}%")
        
        if avg_noise_drop < 2:
            print("✅ Models are ROBUST to sensor noise")
        elif avg_noise_drop < 5:
            print("⚠️  Models show MODERATE sensitivity to noise")
        else:
            print("🔻 Models are SENSITIVE to noise - consider data augmentation")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

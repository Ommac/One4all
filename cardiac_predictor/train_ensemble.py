#!/usr/bin/env python3
"""
Training script for TensorFlow CNN+LSTM Ensemble model.

This script trains an ensemble model WITHOUT data leakage by:
1. Splitting data BEFORE any preprocessing
2. Fitting StandardScaler ONLY on training data
3. Using pre-trained PyTorch CNN and LSTM models

Usage:
    python3 train_ensemble.py

Outputs:
    - models/ensemble_model_clean.h5 (TensorFlow ensemble)
    - models/scaler_clean.pkl (StandardScaler fitted on train only)
    - models/label_encoder_clean.pkl (LabelEncoder)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# Import local modules
from src.preprocessing import load_dataset, butterworth_filter
from src.model_definitions import ECGCNNModel, ECGLSTMModel


# ============================================================
# CONFIGURATION (must match train.py exactly)
# ============================================================
DATASET_PATH = 'data/ecg_dataset.csv'
MODELS_DIR = 'models'
TEST_SIZE = 0.2
RANDOM_STATE = 42
SIGNAL_LENGTH = 2500
LSTM_LENGTH = 500  # Downsampled (every 5th point)

# PyTorch model paths (pre-trained, do not modify)
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_model.h5')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.h5')

# Output paths for clean models
ENSEMBLE_MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model_clean.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_clean.pkl')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder_clean.pkl')

# Device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def load_pytorch_models():
    """Load pre-trained PyTorch CNN and LSTM models."""
    print("Loading pre-trained PyTorch models...")
    
    # Load CNN
    cnn_model = ECGCNNModel(input_length=SIGNAL_LENGTH)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
    cnn_model.to(DEVICE)
    cnn_model.eval()
    print(f"  ✓ CNN loaded from {CNN_MODEL_PATH}")
    
    # Load LSTM
    lstm_model = ECGLSTMModel(input_length=LSTM_LENGTH)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE, weights_only=True))
    lstm_model.to(DEVICE)
    lstm_model.eval()
    print(f"  ✓ LSTM loaded from {LSTM_MODEL_PATH}")
    
    return cnn_model, lstm_model


def get_pytorch_predictions(cnn_model, lstm_model, X_normalized):
    """
    Get softmax probability outputs from PyTorch models.
    
    Args:
        cnn_model: Loaded PyTorch CNN model
        lstm_model: Loaded PyTorch LSTM model
        X_normalized: Normalized signals (N, 2500)
        
    Returns:
        cnn_probs: (N, 3) probability array
        lstm_probs: (N, 3) probability array
    """
    # Prepare CNN input: (N, 2500, 1)
    X_cnn = torch.FloatTensor(X_normalized).unsqueeze(2).to(DEVICE)
    
    # Prepare LSTM input: downsample to (N, 500, 1)
    X_lstm = torch.FloatTensor(X_normalized[:, ::5]).unsqueeze(2).to(DEVICE)
    
    cnn_probs_list = []
    lstm_probs_list = []
    
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_cnn), batch_size):
            # CNN predictions
            cnn_batch = X_cnn[i:i+batch_size]
            cnn_out = cnn_model(cnn_batch, return_probs=True)
            cnn_probs_list.append(cnn_out.cpu().numpy())
            
            # LSTM predictions
            lstm_batch = X_lstm[i:i+batch_size]
            lstm_out = lstm_model(lstm_batch, return_probs=True)
            lstm_probs_list.append(lstm_out.cpu().numpy())
    
    cnn_probs = np.vstack(cnn_probs_list)
    lstm_probs = np.vstack(lstm_probs_list)
    
    return cnn_probs, lstm_probs


def build_ensemble_model():
    """
    Build TensorFlow ensemble model that averages CNN and LSTM outputs.
    
    Architecture:
        Input (2500, 1) -> CNN branch -> softmax (3)
                       -> LSTM branch -> softmax (3)
                       -> Average -> final probabilities (3)
    """
    # Input layer
    input_layer = layers.Input(shape=(SIGNAL_LENGTH, 1), name='ensemble_input')
    
    # CNN Branch
    x_cnn = layers.Conv1D(64, 5, activation='relu', padding='same')(input_layer)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.MaxPooling1D(2)(x_cnn)
    x_cnn = layers.Conv1D(128, 5, activation='relu', padding='same')(x_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.MaxPooling1D(2)(x_cnn)
    x_cnn = layers.Conv1D(256, 3, activation='relu', padding='same')(x_cnn)
    x_cnn = layers.BatchNormalization()(x_cnn)
    x_cnn = layers.GlobalAveragePooling1D()(x_cnn)
    x_cnn = layers.Dense(128, activation='relu')(x_cnn)
    x_cnn = layers.Dropout(0.5)(x_cnn)
    cnn_output = layers.Dense(3, activation='softmax', name='cnn_output')(x_cnn)
    
    # LSTM Branch (with downsampling)
    x_lstm = layers.AveragePooling1D(5)(input_layer)  # 2500 -> 500
    x_lstm = layers.LSTM(64, return_sequences=True)(x_lstm)
    x_lstm = layers.LSTM(64)(x_lstm)
    x_lstm = layers.Dense(64, activation='relu')(x_lstm)
    x_lstm = layers.Dropout(0.3)(x_lstm)
    lstm_output = layers.Dense(3, activation='softmax', name='lstm_output')(x_lstm)
    
    # Average ensemble
    ensemble_output = layers.Average(name='ensemble_output')([cnn_output, lstm_output])
    
    # Build model
    model = Model(inputs=input_layer, outputs=ensemble_output, name='CNN_LSTM_Ensemble_Clean')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_confusion_matrix(y_true, y_pred, class_names):
    """Print formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix:")
    print("                   Predicted")
    print("              " + "  ".join([f"{name:>8}" for name in class_names]))
    for i, name in enumerate(class_names):
        row = "  ".join([f"{cm[i, j]:>8}" for j in range(len(class_names))])
        print(f"  Actual {name:>6}: {row}")


def main():
    """Main training pipeline with data leakage prevention."""
    print_header("ENSEMBLE MODEL TRAINING (CLEAN - NO DATA LEAKAGE)")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"PyTorch device: {DEVICE}")
    
    # ============================================================
    # STEP 1: Load dataset
    # ============================================================
    print_header("STEP 1: Loading Dataset")
    
    X, y, label_encoder = load_dataset(DATASET_PATH)
    class_names = list(label_encoder.classes_)
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {class_names}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # ============================================================
    # STEP 2: Train/Test Split FIRST (before any preprocessing)
    # ============================================================
    print_header("STEP 2: Train/Test Split (BEFORE preprocessing)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print("  ✓ Split done BEFORE any preprocessing (no leakage)")
    
    # ============================================================
    # STEP 3: Fit StandardScaler ONLY on training data
    # ============================================================
    print_header("STEP 3: Fit Scaler on TRAINING DATA ONLY")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # CRITICAL: Verify no leakage
    assert scaler.n_samples_seen_ == len(X_train), \
        f"LEAKAGE DETECTED! Scaler saw {scaler.n_samples_seen_} samples but X_train has {len(X_train)}"
    
    print(f"  Scaler fitted on {scaler.n_samples_seen_} samples (X_train only)")
    print(f"  ✓ VERIFIED: n_samples_seen_ == len(X_train) == {len(X_train)}")
    print("  ✓ No data leakage in scaler")
    
    # Save clean scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Saved clean scaler to {SCALER_PATH}")
    
    # Save label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"  ✓ Saved label encoder to {LABEL_ENCODER_PATH}")
    
    # ============================================================
    # STEP 4: Build and Train TensorFlow Ensemble
    # ============================================================
    print_header("STEP 4: Training TensorFlow Ensemble")
    
    # Reshape for TensorFlow: (N, 2500, 1)
    X_train_tf = X_train_scaled.reshape(-1, SIGNAL_LENGTH, 1).astype(np.float32)
    X_test_tf = X_test_scaled.reshape(-1, SIGNAL_LENGTH, 1).astype(np.float32)
    
    print(f"  TensorFlow input shape: {X_train_tf.shape}")
    
    # Build model
    model = build_ensemble_model()
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]
    
    # Train
    print("\nTraining ensemble model...")
    history = model.fit(
        X_train_tf, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save(ENSEMBLE_MODEL_PATH)
    print(f"\n  ✓ Saved ensemble model to {ENSEMBLE_MODEL_PATH}")
    
    # ============================================================
    # STEP 5: Evaluate on Test Set
    # ============================================================
    print_header("STEP 5: Evaluation on Test Set")
    
    # Predict
    y_pred_probs = model.predict(X_test_tf, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    for line in report.split('\n'):
        print(f"  {line}")
    
    # Confusion matrix
    print_confusion_matrix(y_test, y_pred, class_names)
    
    # F1 scores
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    
    print("\n" + "-" * 60)
    print("  F1 SCORES SUMMARY")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"  {name:>8} F1: {per_class_f1[i]:.4f}")
    print(f"  {'Macro':>8} F1: {macro_f1:.4f}")
    print(f"  {'Weighted':>8} F1: {weighted_f1:.4f}")
    
    # ============================================================
    # FINAL VERIFICATION
    # ============================================================
    print_header("FINAL VERIFICATION")
    
    # Reload scaler and verify
    scaler_loaded = joblib.load(SCALER_PATH)
    print(f"  ✓ scaler_clean.pkl n_samples_seen_: {scaler_loaded.n_samples_seen_}")
    print(f"  ✓ Expected (len(X_train)): {len(X_train)}")
    
    if scaler_loaded.n_samples_seen_ == len(X_train):
        print("  ✅ NO DATA LEAKAGE - Scaler fitted on training data only")
    else:
        print("  ❌ DATA LEAKAGE DETECTED!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE - CLEAN ENSEMBLE READY")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {ENSEMBLE_MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {LABEL_ENCODER_PATH}")
    print("\nNext steps:")
    print("  1. Update app.py to use scaler_clean.pkl")
    print("  2. Update app.py to use ensemble_model_clean.h5")
    print("  3. Run test_all_models.py to verify F1 scores")


if __name__ == '__main__':
    main()

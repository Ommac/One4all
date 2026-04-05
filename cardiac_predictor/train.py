"""
Training script for cardiac arrhythmia classification models.
Trains CNN and LSTM models using the ECG dataset.

Usage:
    python3 train.py

Models are saved to:
    - models/cnn_model.h5 (PyTorch CNN state dict)
    - models/lstm_model.h5 (PyTorch LSTM state dict)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import local modules
from src.preprocessing import load_dataset, preprocess_for_nn
from src.model_definitions import ECGCNNModel, ECGLSTMModel


# Configuration
DATASET_PATH = 'data/ecg_dataset.csv'
MODELS_DIR = 'models'
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 10
BATCH_SIZE = 32
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_models_dir():
    """Create models directory if it doesn't exist."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")


def remove_duplicates(X, y):
    """
    Remove duplicate rows from the dataset.
    
    Args:
        X: Feature array
        y: Label array
        
    Returns:
        X_unique, y_unique with duplicates removed
    """
    # Convert to tuple for hashing
    seen = set()
    unique_indices = []
    
    for i in range(len(X)):
        # Create hash of the signal (first 100 + last 100 values for efficiency)
        sig_hash = hash(tuple(X[i, :100].tolist() + X[i, -100:].tolist()))
        if sig_hash not in seen:
            seen.add(sig_hash)
            unique_indices.append(i)
    
    n_duplicates = len(X) - len(unique_indices)
    if n_duplicates > 0:
        print(f"  Removed {n_duplicates} duplicate rows")
    
    return X[unique_indices], y[unique_indices]


def validate_no_overlap(X_train, X_test):
    """
    Verify no overlap between train and test sets.
    
    Args:
        X_train: Training data
        X_test: Test data
        
    Returns:
        True if no overlap, raises error otherwise
    """
    train_hashes = set(hash(tuple(x[:100].tolist() + x[-100:].tolist())) for x in X_train)
    test_hashes = set(hash(tuple(x[:100].tolist() + x[-100:].tolist())) for x in X_test)
    
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(f"Data leakage detected! {len(overlap)} samples overlap between train/test")
    
    print("  ✓ No overlap between train and test sets")
    return True


def normalize_data(X_train, X_test):
    """
    Normalize data using StandardScaler fitted ONLY on training data.
    This prevents data leakage from test set.
    
    Args:
        X_train: Training signals (N_train, timesteps)
        X_test: Test signals (N_test, timesteps)
        
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def print_confusion_matrix(y_true, y_pred, label_names):
    """Print formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("            " + "  ".join([f"{name:>8}" for name in label_names]))
    for i, name in enumerate(label_names):
        row = "  ".join([f"{cm[i, j]:>8}" for j in range(len(label_names))])
        print(f"Actual {name:>6}: {row}")


def explain_high_f1(f1_score_val, model_name):
    """Explain why F1 score might be high."""
    if f1_score_val > 0.95:
        print(f"\n⚠️  {model_name} F1={f1_score_val:.4f} (>0.95)")
        print("   Possible reasons for high performance:")
        print("   1. ECG signals have distinct patterns for each arrhythmia type")
        print("   2. Dataset may have clear class separation")
        print("   3. Synthetic/curated data with minimal noise")
        print("   4. Consider testing on real-world clinical data for validation")


def train_cnn(X_train, y_train, X_test, y_test, label_names):
    """
    Train CNN model for ECG classification.
    Normalizes data using training statistics only (no leakage).
    
    Args:
        X_train: Training signals (N, timesteps)
        y_train: Training labels (N,)
        X_test: Test signals (N, timesteps)
        y_test: Test labels (N,)
        label_names: List of class names for report
        
    Returns:
        Trained PyTorch model, macro F1 score
    """
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    print(f"Using device: {DEVICE}")
    
    # Normalize using training data only (prevents leakage)
    X_train_norm, X_test_norm, _ = normalize_data(X_train, X_test)
    print("  ✓ Data normalized (fitted on training data only)")
    
    # Reshape for CNN: (samples, timesteps, channels)
    X_train_3d = preprocess_for_nn(X_train_norm)
    X_test_3d = preprocess_for_nn(X_test_norm)
    
    print(f"CNN input shape: {X_train_3d.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_3d)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_3d)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Build model
    input_length = X_train_3d.shape[1]
    model = ECGCNNModel(input_length=input_length, num_classes=3).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X, return_probs=False)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(DEVICE)
        outputs = model(X_test_device, return_probs=True)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    print("\n--- CNN Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    print_confusion_matrix(y_test, y_pred, label_names)
    explain_high_f1(macro_f1, "CNN")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'cnn_model.h5')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    return model, macro_f1


def train_lstm(X_train, y_train, X_test, y_test, label_names):
    """
    Train LSTM model for ECG classification.
    Uses downsampled signals for faster training.
    Normalizes data using training statistics only (no leakage).
    
    Args:
        X_train: Training signals (N, timesteps)
        y_train: Training labels (N,)
        X_test: Test signals (N, timesteps)
        y_test: Test labels (N,)
        label_names: List of class names for report
        
    Returns:
        Trained PyTorch model, macro F1 score
    """
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    print(f"Using device: {DEVICE}")
    
    # Downsample signals for LSTM (every 5th point: 2500 -> 500)
    downsample_factor = 5
    X_train_ds = X_train[:, ::downsample_factor]
    X_test_ds = X_test[:, ::downsample_factor]
    
    # Normalize using training data only (prevents leakage)
    X_train_norm, X_test_norm, _ = normalize_data(X_train_ds, X_test_ds)
    print("  ✓ Data normalized (fitted on training data only)")
    
    # Reshape for LSTM: (samples, timesteps, channels)
    X_train_3d = np.expand_dims(X_train_norm, axis=-1).astype(np.float32)
    X_test_3d = np.expand_dims(X_test_norm, axis=-1).astype(np.float32)
    
    print(f"LSTM input shape (downsampled): {X_train_3d.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_3d)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_3d)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Build model
    input_length = X_train_3d.shape[1]
    model = ECGLSTMModel(input_length=input_length, num_classes=3).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X, return_probs=False)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(DEVICE)
        outputs = model(X_test_device, return_probs=True)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    print("\n--- LSTM Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=label_names))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    print_confusion_matrix(y_test, y_pred, label_names)
    explain_high_f1(macro_f1, "LSTM")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    return model, macro_f1


def main():
    """Main training pipeline with data leakage prevention."""
    print("="*60)
    print("CARDIAC ARRHYTHMIA CLASSIFIER - TRAINING PIPELINE")
    print("="*60)
    print("\n🔒 Data Leakage Prevention Enabled")
    
    # Ensure models directory exists
    ensure_models_dir()
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    try:
        X, y, label_encoder = load_dataset(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Please ensure the dataset exists at: {DATASET_PATH}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Get label names for classification report
    label_names = list(label_encoder.classes_)
    print(f"Classes: {label_names}")
    
    # Step 2: Remove duplicates BEFORE splitting
    print("\n[Step 2] Data Validation...")
    print("  Checking for duplicate rows...")
    X, y = remove_duplicates(X, y)
    print(f"  Dataset size after dedup: {len(X)} samples")
    
    # Step 3: Split data BEFORE any preprocessing (prevents leakage)
    print("\n[Step 3] Splitting dataset (80/20 stratified)...")
    print("  ⚠️  Split happens BEFORE preprocessing to prevent leakage")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Validate no overlap
    validate_no_overlap(X_train, X_test)
    
    # Step 4: Train neural network models
    print("\n[Step 4] Training models...")
    print("  Note: All normalization is fitted on TRAINING data only")
    
    f1_scores = {}

    # Train CNN
    _, f1_scores['CNN'] = train_cnn(X_train, y_train, X_test, y_test, label_names)
    
    # Train LSTM
    _, f1_scores['LSTM'] = train_lstm(X_train, y_train, X_test, y_test, label_names)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModels saved to: {MODELS_DIR}/")
    print("  - cnn_model.h5 (CNN)")
    print("  - lstm_model.h5 (LSTM)")
    
    print("\n📊 F1 Score Summary:")
    for model_name, f1 in f1_scores.items():
        status = "⚠️ HIGH" if f1 > 0.95 else "✓"
        print(f"  {model_name}: {f1:.4f} {status}")
    
    print("\n🔒 Data Leakage Prevention Summary:")
    print("  ✓ Train/test split done BEFORE preprocessing")
    print("  ✓ Normalization fitted on training data only")
    print("  ✓ No duplicate rows")
    print("  ✓ No overlap between train/test sets")
    print("="*60)


if __name__ == '__main__':
    main()

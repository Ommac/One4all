"""
Training script for cardiac arrhythmia classification models.
Trains XGBoost, CNN (ECGNet), and LSTM (ECGLSTMNet) models.
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

from src.preprocessing import load_dataset, butterworth_filter
from src.feature_extraction import extract_features
from src.model_definitions import ECGNet, ECGLSTMNet


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> XGBClassifier:
    """
    Train XGBoost classifier on extracted features.
    
    Args:
        X_train: Training signals of shape (N_train, 960)
        y_train: Training labels of shape (N_train,)
        X_test: Test signals of shape (N_test, 960)
        y_test: Test labels of shape (N_test,)
        
    Returns:
        Trained XGBClassifier model
    """
    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)
    
    # Extract features from all signals
    print("Extracting features...")
    X_train_features = np.array([extract_features(sig) for sig in X_train])
    X_test_features = np.array([extract_features(sig) for sig in X_test])
    
    print(f"Training features shape: {X_train_features.shape}")
    print(f"Test features shape: {X_test_features.shape}")
    
    # Initialize and train XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42
    )
    
    print("Training XGBoost model...")
    xgb_model.fit(X_train_features, y_train)
    
    # Evaluate on test set
    y_pred = xgb_model.predict(X_test_features)
    
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "AFib", "VFib"]))
    
    # Calculate and print specific metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    vfib_f1 = f1_score(y_test, y_pred, labels=[2], average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"VFib F1 Score: {vfib_f1:.4f}")
    
    # Save model
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("XGBoost model saved to models/xgb_model.pkl")
    
    return xgb_model


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> ECGNet:
    """
    Train CNN (ECGNet) model.
    
    Args:
        X_train: Training signals of shape (N_train, 960)
        y_train: Training labels of shape (N_train,)
        X_test: Test signals of shape (N_test, 960)
        y_test: Test labels of shape (N_test,)
        
    Returns:
        Trained ECGNet model
    """
    print("\n" + "="*60)
    print("Training CNN (ECGNet)")
    print("="*60)
    
    # Reshape data for CNN: (N, 960) → (N, 1, 960)
    X_train_cnn = np.expand_dims(X_train, axis=1)
    X_test_cnn = np.expand_dims(X_test, axis=1)
    
    print(f"CNN input shape: {X_train_cnn.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_cnn)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_cnn)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ECGNet().to(device)
    
    # Class weights: give more importance to VFib (class 2)
    class_weights = torch.FloatTensor([1.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 30
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        outputs = model(X_test_device)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
    
    print("\nCNN Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "AFib", "VFib"]))
    
    # Calculate and print specific metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    vfib_f1 = f1_score(y_test, y_pred, labels=[2], average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"VFib F1 Score: {vfib_f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/cnn_model.pt')
    print("CNN model saved to models/cnn_model.pt")
    
    return model


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray) -> ECGLSTMNet:
    """
    Train LSTM (ECGLSTMNet) model.
    
    Args:
        X_train: Training signals of shape (N_train, 960)
        y_train: Training labels of shape (N_train,)
        X_test: Test signals of shape (N_test, 960)
        y_test: Test labels of shape (N_test,)
        
    Returns:
        Trained ECGLSTMNet model
    """
    print("\n" + "="*60)
    print("Training LSTM (ECGLSTMNet)")
    print("="*60)
    
    # Reshape data for LSTM: (N, 960) → (N, 960, 1)
    X_train_lstm = np.expand_dims(X_train, axis=2)
    X_test_lstm = np.expand_dims(X_test, axis=2)
    
    print(f"LSTM input shape: {X_train_lstm.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_lstm)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_lstm)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ECGLSTMNet().to(device)
    
    # Class weights: give more importance to VFib (class 2)
    class_weights = torch.FloatTensor([1.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 30
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        outputs = model(X_test_device)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.cpu().numpy()
    
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "AFib", "VFib"]))
    
    # Calculate and print specific metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    vfib_f1 = f1_score(y_test, y_pred, labels=[2], average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"VFib F1 Score: {vfib_f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/lstm_model.pt')
    print("LSTM model saved to models/lstm_model.pt")
    
    return model


def main():
    """Main training pipeline."""
    print("="*60)
    print("Cardiac Arrhythmia Classifier Training")
    print("="*60)
    
    # Step 1: Load dataset
    print("\nLoading dataset...")
    X, y = load_dataset('data/ecg_dataset.csv')
    print(f"Loaded {len(X)} samples")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution: Normal={np.sum(y==0)}, AFib={np.sum(y==1)}, VFib={np.sum(y==2)}")
    
    # Step 2: Apply Butterworth filter to all signals
    print("\nApplying Butterworth bandpass filter...")
    X_clean = np.array([butterworth_filter(sig) for sig in X])
    print(f"Filtered X shape: {X_clean.shape}")
    
    # Step 3: Train/test split (80/20, stratified)
    print("\nSplitting data (80/20 stratified split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 4: Train all models
    train_xgboost(X_train, y_train, X_test, y_test)
    train_cnn(X_train, y_train, X_test, y_test)
    train_lstm(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("Models saved to models/ directory")
    print("="*60)


if __name__ == '__main__':
    main()

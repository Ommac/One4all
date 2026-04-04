"""
PyTorch model definitions for ECG classification.
Contains CNN (ECGNet) and LSTM (ECGLSTMNet) architectures.
"""

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    """
    Convolutional Neural Network for ECG classification.
    
    Architecture:
    - 3 convolutional blocks with BatchNorm, ReLU, and MaxPooling
    - Fully connected layers with dropout for classification
    
    Input shape: (batch, 1, 960)
    Output shape: (batch, 3) - raw logits for 3 classes
    """
    
    def __init__(self):
        """Initialize the ECGNet CNN architecture."""
        super(ECGNet, self).__init__()
        
        # Convolutional Block 1
        # Input: (batch, 1, 960) → Output: (batch, 32, 480)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Convolutional Block 2
        # Input: (batch, 32, 480) → Output: (batch, 64, 240)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Convolutional Block 3
        # Input: (batch, 64, 240) → Output: (batch, 128, 120)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        # Input: 128 * 120 = 15360 features
        self.fc1 = nn.Linear(128 * 120, 256)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, 960)
            
        Returns:
            Raw logits tensor of shape (batch, 3)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten and FC layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ECGLSTMNet(nn.Module):
    """
    LSTM-based Neural Network for ECG classification.
    
    Architecture:
    - 2-layer LSTM with dropout
    - Fully connected layers for classification
    
    Input shape: (batch, 960, 1)
    Output shape: (batch, 3) - raw logits for 3 classes
    """
    
    def __init__(self):
        """Initialize the ECGLSTMNet LSTM architecture."""
        super(ECGLSTMNet, self).__init__()
        
        # LSTM layer
        # Input: (batch, 960, 1) → Output: (batch, 960, 128)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 960, 1)
            
        Returns:
            Raw logits tensor of shape (batch, 3)
        """
        # LSTM forward pass
        # lstm_out shape: (batch, 960, 128)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last timestep
        # Shape: (batch, 128)
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

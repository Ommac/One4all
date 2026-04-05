"""
PyTorch model definitions for ECG classification.
Contains CNN and LSTM architectures.
"""

import torch
import torch.nn as nn


class ECGCNNModel(nn.Module):
    """
    1D CNN model for ECG classification.
    
    Architecture:
    - 3 Conv1D blocks with BatchNorm, ReLU, and MaxPooling
    - Global Average Pooling for variable-length sequences
    - Dense layers with dropout for classification
    - Softmax output for 3 classes
    
    Input shape: (batch, timesteps, 1) - will be transposed internally
    Output shape: (batch, 3)
    """
    
    def __init__(self, input_length: int, num_classes: int = 3):
        """
        Initialize CNN model.
        
        Args:
            input_length: Length of input signal (e.g., 2500)
            num_classes: Number of output classes (default 3)
        """
        super(ECGCNNModel, self).__init__()
        
        # Conv Block 1: 32 filters, kernel size 5
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv Block 2: 64 filters, kernel size 5
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Conv Block 3: 128 filters, kernel size 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 256)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, timesteps, 1)
            return_probs: If True, return softmax probabilities; else return logits
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Transpose from (batch, timesteps, 1) to (batch, 1, timesteps) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        
        # Dense layers
        x = self.dropout(self.relu_fc(self.fc1(x)))
        x = self.fc2(x)
        
        if return_probs:
            x = self.softmax(x)
        
        return x


class ECGLSTMModel(nn.Module):
    """
    LSTM model for ECG classification.
    
    Architecture:
    - 2 LSTM layers with dropout
    - Dense layers for classification
    - Softmax output for 3 classes
    
    Input shape: (batch, timesteps, 1)
    Output shape: (batch, 3)
    """
    
    def __init__(self, input_length: int, num_classes: int = 3):
        """
        Initialize LSTM model.
        
        Args:
            input_length: Length of input signal (e.g., 2500)
            num_classes: Number of output classes (default 3)
        """
        super(ECGLSTMModel, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Dense layers
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, timesteps, 1)
            return_probs: If True, return softmax probabilities; else return logits
            
        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm1(x)
        
        # Take output from last timestep
        x = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        if return_probs:
            x = self.softmax(x)
        
        return x

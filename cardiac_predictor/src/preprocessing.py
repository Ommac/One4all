"""
Preprocessing module for ECG signal data.
Contains Butterworth bandpass filter and dataset loading utilities.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
import ast
import json
from typing import Tuple


def butterworth_filter(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to an ECG signal.
    
    Filters the signal to keep frequencies between 0.5Hz and 40Hz,
    which is the typical range for cardiac electrical activity.
    
    Args:
        signal: 1D numpy array of ECG signal values
        fs: Sampling rate in Hz (default 360)
        
    Returns:
        Filtered 1D numpy array of same shape as input
    """
    # Filter parameters
    lowcut = 0.5      # Low frequency cutoff in Hz
    highcut = 40.0    # High frequency cutoff in Hz
    order = 4         # Filter order
    
    # Calculate normalized frequencies (Nyquist frequency = fs/2)
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Clamp high frequency to valid range
    high = min(high, 0.99)
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def parse_signal_string(signal_str: str) -> np.ndarray:
    """
    Parse a signal stored as a string representation of a list.
    
    Handles formats like "[0.031, 0.038, ...]" using ast.literal_eval
    or json.loads as fallback, or comma-separated values.
    
    Args:
        signal_str: String representation of signal list
        
    Returns:
        1D numpy array of signal values
    """
    # Try ast.literal_eval first (handles Python list syntax like "[0.1, 0.2, ...]")
    try:
        signal_list = ast.literal_eval(signal_str)
        return np.array(signal_list, dtype=np.float32)
    except (ValueError, SyntaxError):
        pass
    
    # Try json.loads as fallback
    try:
        signal_list = json.loads(signal_str)
        return np.array(signal_list, dtype=np.float32)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try comma-separated values (no brackets)
    try:
        values = [float(x.strip()) for x in signal_str.split(',')]
        return np.array(values, dtype=np.float32)
    except ValueError:
        pass
    
    raise ValueError(f"Could not parse signal string: {signal_str[:50]}...")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load ECG dataset from CSV file.
    
    Expects CSV with columns:
    - 'label': String labels (Normal, AFib, VFib)
    - 'signal': ECG signal as string list "[0.031, 0.038, ...]"
    
    Args:
        filepath: Path to the CSV file containing ECG data
        
    Returns:
        Tuple of (X, y, label_encoder) where:
            X: numpy array of shape (N, signal_length) containing signal data
            y: numpy array of shape (N,) containing encoded integer labels
            label_encoder: Fitted LabelEncoder for inverse transform
            
    Raises:
        FileNotFoundError: If filepath does not exist
        ValueError: If required columns are missing
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Validate required columns
    if 'label' not in df.columns:
        raise ValueError("CSV must have 'label' column")
    if 'signal' not in df.columns:
        raise ValueError("CSV must have 'signal' column")
    
    print(f"Loaded {len(df)} samples from {filepath}")
    
    # Parse all signals
    signals = []
    for idx, row in df.iterrows():
        signal_str = row['signal']
        try:
            signal = parse_signal_string(signal_str)
            signals.append(signal)
        except ValueError as e:
            print(f"Warning: Skipping row {idx} - {e}")
            continue
    
    # Stack signals into numpy array
    X = np.stack(signals, axis=0).astype(np.float32)
    
    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'].values)
    
    print(f"Signal shape: {X.shape}")
    print(f"Labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, label_encoder


def preprocess_for_nn(X: np.ndarray) -> np.ndarray:
    """
    Prepare data for CNN/LSTM (3D: samples x timesteps x channels).
    
    Args:
        X: Signal array of shape (samples, timesteps)
        
    Returns:
        3D array of shape (samples, timesteps, 1)
    """
    # Add channel dimension for Keras Conv1D/LSTM
    return np.expand_dims(X, axis=-1)

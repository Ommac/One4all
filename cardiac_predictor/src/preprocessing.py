"""
Preprocessing module for ECG signal data.
Contains Butterworth bandpass filter and dataset loading utilities.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import ast
import json
from typing import Tuple


def butterworth_filter(signal: np.ndarray) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to an ECG signal.
    
    Filters the signal to keep frequencies between 0.5Hz and 40Hz,
    which is the typical range for cardiac electrical activity.
    
    Args:
        signal: 1D numpy array of ECG signal values (960 points)
        
    Returns:
        Filtered 1D numpy array of same shape as input
    """
    # Filter parameters
    lowcut = 0.5      # Low frequency cutoff in Hz
    highcut = 40.0    # High frequency cutoff in Hz
    fs = 360          # Sampling rate in Hz
    order = 4         # Filter order
    
    # Calculate normalized frequencies (Nyquist frequency = fs/2)
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter using filtfilt for zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def _parse_signal_string(signal_str: str) -> np.ndarray:
    """
    Parse a signal stored as a string representation of a list.
    
    Handles formats like "[0.031, 0.038, ...]" using ast.literal_eval
    or json.loads as fallback.
    
    Args:
        signal_str: String representation of signal list
        
    Returns:
        1D numpy array of signal values
    """
    try:
        # Try ast.literal_eval first (handles Python list syntax)
        signal_list = ast.literal_eval(signal_str)
        return np.array(signal_list, dtype=np.float32)
    except (ValueError, SyntaxError):
        pass
    
    try:
        # Try json.loads as fallback
        signal_list = json.loads(signal_str)
        return np.array(signal_list, dtype=np.float32)
    except json.JSONDecodeError:
        pass
    
    raise ValueError(f"Could not parse signal string: {signal_str[:50]}...")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG dataset from CSV file, handling both signal storage formats.
    
    Format A: signal column contains a string like "[0.031, 0.038, ...]"
    Format B: signal values spread across columns (col1, col2, ... col960)
    
    Args:
        filepath: Path to the CSV file containing ECG data
        
    Returns:
        Tuple of (X, y) where:
            X: numpy array of shape (N, 960) containing signal data
            y: numpy array of shape (N,) containing integer labels (0, 1, or 2)
    """
    # Label mapping: string labels to integer encoding
    label_map = {
        "Normal": 0,
        "AFib": 1,
        "VFib": 2
    }
    
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Initialize lists to collect signals and labels
    signals = []
    labels = []
    
    # Process each row
    for idx, row in df.iterrows():
        # Extract label (always in 'label' column)
        label_str = row['label']
        label_int = label_map[label_str]
        labels.append(label_int)
        
        # Check if 'signal' column exists and contains string data (Format A)
        if 'signal' in df.columns:
            signal_value = row['signal']
            
            # Check if it's a string (Format A)
            if isinstance(signal_value, str):
                signal = _parse_signal_string(signal_value)
                signals.append(signal)
                continue
            
            # If signal column exists but value is numeric, might be first value of spread format
            # Fall through to Format B handling
        
        # Format B: signal spread across multiple columns
        # Get all columns except 'label' and optionally 'signal'
        numeric_cols = [col for col in df.columns if col != 'label']
        if 'signal' in numeric_cols and isinstance(row['signal'], str):
            numeric_cols.remove('signal')
        
        # Extract numeric values from all columns after label
        signal_values = []
        for col in numeric_cols:
            try:
                val = float(row[col])
                signal_values.append(val)
            except (ValueError, TypeError):
                continue
        
        signal = np.array(signal_values, dtype=np.float32)
        
        # Ensure we have exactly 960 points
        if len(signal) != 960:
            # If signal column had partial data, try combining
            if 'signal' in df.columns and not isinstance(row['signal'], str):
                try:
                    first_val = float(row['signal'])
                    signal = np.concatenate([[first_val], signal])
                except (ValueError, TypeError):
                    pass
        
        signals.append(signal)
    
    # Convert to numpy arrays
    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    return X, y

"""
Feature extraction module for ECG signals.
Extracts statistical features for use with XGBoost classifier.
"""

import numpy as np
from scipy.signal import find_peaks


def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract 8 statistical features from a cleaned ECG signal for XGBoost.
    
    Features extracted:
    1. mean: Average value of the signal
    2. std: Standard deviation of the signal
    3. max: Maximum value in the signal
    4. min: Minimum value in the signal
    5. range: Difference between max and min values
    6. percentile_25: 25th percentile of signal values
    7. percentile_75: 75th percentile of signal values
    8. peak_count: Number of peaks detected in the signal
    
    Args:
        signal: 1D numpy array of 960 ECG signal points (cleaned/filtered)
        
    Returns:
        1D numpy array of 8 feature values
    """
    # Feature 1: Mean value
    mean_val = np.mean(signal)
    
    # Feature 2: Standard deviation
    std_val = np.std(signal)
    
    # Feature 3: Maximum value
    max_val = np.max(signal)
    
    # Feature 4: Minimum value
    min_val = np.min(signal)
    
    # Feature 5: Range (max - min)
    range_val = max_val - min_val
    
    # Feature 6: 25th percentile
    percentile_25 = np.percentile(signal, 25)
    
    # Feature 7: 75th percentile
    percentile_75 = np.percentile(signal, 75)
    
    # Feature 8: Peak count (number of local maxima)
    # Use scipy's find_peaks to detect peaks in the signal
    # Set minimum height to mean + 0.5*std to filter noise
    height_threshold = mean_val + 0.5 * std_val
    peaks, _ = find_peaks(signal, height=height_threshold)
    peak_count = len(peaks)
    
    # Combine all features into a single array
    features = np.array([
        mean_val,
        std_val,
        max_val,
        min_val,
        range_val,
        percentile_25,
        percentile_75,
        peak_count
    ], dtype=np.float32)
    
    return features

"""
Ensemble prediction module for cardiac arrhythmia classification.
Combines predictions from CNN and LSTM models.
"""

from typing import Dict, List


def ensemble_predict(
    cnn_pred: int,
    lstm_pred: int,
    cnn_probs: List[float]
) -> Dict:
    """
    Combine predictions from CNN and LSTM models using ensemble rules.
    
    Ensemble Rules:
    1. Safety First: If ANY model predicts VFib (2), final prediction is VFib
    2. Otherwise: If both models agree, use that prediction
    3. Tiebreak: CNN prediction wins (more reliable based on testing)
    
    Args:
        cnn_pred: CNN prediction (0=AFib, 1=Normal, 2=VFib)
        lstm_pred: LSTM prediction (0=AFib, 1=Normal, 2=VFib)
        cnn_probs: CNN probability distribution [P(AFib), P(Normal), P(VFib)]
        
    Returns:
        Dictionary containing:
            - prediction: String label ("Normal", "AFib", or "VFib")
            - confidence: Float confidence score from CNN probabilities
            - probabilities: Dict with probability for each class
    """
    # Label mapping for output (alphabetical: AFib=0, Normal=1, VFib=2)
    label_map = {0: "AFib", 1: "Normal", 2: "VFib"}
    
    # Collect all predictions
    predictions = [cnn_pred, lstm_pred]
    
    # Rule 1: Safety First - if ANY model predicts VFib, output VFib
    if 2 in predictions:
        final_pred = 2
    # Rule 2: Both models agree
    elif cnn_pred == lstm_pred:
        final_pred = cnn_pred
    else:
        # Rule 3: Tiebreak - CNN wins (more reliable)
        final_pred = cnn_pred
    
    # Calculate confidence from CNN probabilities
    confidence = cnn_probs[final_pred]
    
    # Build probabilities dictionary
    probabilities = {
        "AFib": float(cnn_probs[0]),
        "Normal": float(cnn_probs[1]),
        "VFib": float(cnn_probs[2])
    }
    
    return {
        "prediction": label_map[final_pred],
        "confidence": float(confidence),
        "probabilities": probabilities
    }

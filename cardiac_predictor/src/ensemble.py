"""
Ensemble prediction module for cardiac arrhythmia classification.
Combines predictions from XGBoost, CNN, and LSTM models.
"""

from typing import Dict, List


def ensemble_predict(
    xgb_pred: int,
    cnn_pred: int,
    lstm_pred: int,
    cnn_probs: List[float]
) -> Dict:
    """
    Combine predictions from three models using ensemble rules.
    
    Ensemble Rules:
    1. Safety First: If ANY model predicts VFib (2), final prediction is VFib
    2. Otherwise: Use majority voting among the three predictions
    3. Tiebreak: CNN prediction wins in case of a three-way tie
    
    Args:
        xgb_pred: XGBoost prediction (0=Normal, 1=AFib, 2=VFib)
        cnn_pred: CNN prediction (0=Normal, 1=AFib, 2=VFib)
        lstm_pred: LSTM prediction (0=Normal, 1=AFib, 2=VFib)
        cnn_probs: CNN probability distribution [P(Normal), P(AFib), P(VFib)]
        
    Returns:
        Dictionary containing:
            - prediction: String label ("Normal", "AFib", or "VFib")
            - confidence: Float confidence score from CNN probabilities
            - probabilities: Dict with probability for each class
    """
    # Label mapping for output
    label_map = {0: "Normal", 1: "AFib", 2: "VFib"}
    
    # Collect all predictions
    predictions = [xgb_pred, cnn_pred, lstm_pred]
    
    # Rule 1: Safety First - if ANY model predicts VFib, output VFib
    if 2 in predictions:
        final_pred = 2
    else:
        # Rule 2: Majority voting
        vote_counts = {}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        # Find the maximum vote count
        max_votes = max(vote_counts.values())
        
        # Get all predictions with max votes
        winners = [pred for pred, count in vote_counts.items() if count == max_votes]
        
        if len(winners) == 1:
            # Clear majority winner
            final_pred = winners[0]
        else:
            # Rule 3: Tiebreak - CNN wins
            final_pred = cnn_pred
    
    # Calculate confidence from CNN probabilities
    # Confidence is the probability of the final prediction class
    confidence = cnn_probs[final_pred]
    
    # Build probabilities dictionary
    probabilities = {
        "Normal": float(cnn_probs[0]),
        "AFib": float(cnn_probs[1]),
        "VFib": float(cnn_probs[2])
    }
    
    return {
        "prediction": label_map[final_pred],
        "confidence": float(confidence),
        "probabilities": probabilities
    }

import torch
import random
import numpy as np

def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def accuracy(y_pred, y_true, threshold=0.5):
    """
    Computes the accuracy between the ground truth labels and the predicted labels.

    Args:
        y_true (torch.Tensor): Ground truth binary or multi-class labels, shape (N,) or (N, C)
        y_pred (torch.Tensor): Predicted labels (probabilities), shape (N,) or (N, C)
        threshold (float): Threshold to binarize y_pred for binary classification, default is 0.5

    Returns:
        float: Accuracy
    """

    y_pred = (y_pred > threshold).float()

    # Calculate the number of correct predictions
    correct_predictions = (y_true == y_pred).sum().item()

    # Calculate the total number of instances
    total_instances = y_true.numel()

    # Calculate accuracy
    accuracy = correct_predictions / total_instances

    return accuracy
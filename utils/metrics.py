import numpy as np
import torch

def compute_segmentation_metrics(pred_masks, true_masks, num_classes):
    """
    Computes common segmentation metrics (Pixel Accuracy, IoU, FWIoU).
    Args:
        pred_masks (np.ndarray): Predicted class indices (N, H, W).
        true_masks (np.ndarray): Ground truth class indices (N, H, W).
        num_classes (int): Total number of classes.
    """
    # Flatten the masks for easier comparison
    pred_flat = pred_masks.flatten()
    true_flat = true_masks.flatten()

    # 1. Confusion Matrix
    # Computes the confusion matrix, where C[i, j] is the count of pixels
    # of true class i being predicted as class j.
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((true_flat == i) & (pred_flat == j))

    # Calculate key terms from the confusion matrix
    TP = np.diag(confusion_matrix)             # True Positives
    FP = np.sum(confusion_matrix, axis=0) - TP  # False Positives
    FN = np.sum(confusion_matrix, axis=1) - TP  # False Negatives
    TN = np.sum(confusion_matrix) - (TP + FP + FN) # True Negatives (for binary, but less relevant for multi-class)
    
    # Sum of all pixels
    total_pixels = np.sum(confusion_matrix)

    # 2. Pixel Accuracy
    pixel_accuracy = np.sum(TP) / total_pixels
    
    # 3. Intersection over Union (IoU) per Class
    # IoU = TP / (TP + FP + FN)
    denominator = TP + FP + FN
    # Handle division by zero (classes not present in the test set)
    iou_per_class = np.divide(TP, denominator, out=np.zeros_like(TP, dtype=float), where=denominator != 0)
    m_iou = np.mean(iou_per_class)

    # 4. Frequency Weighted IoU (FWIoU)
    # FWIoU = (TP + FN) / total_pixels * IoU
    freq = np.sum(confusion_matrix, axis=1) / total_pixels
    fwiou = np.sum(freq[freq > 0] * iou_per_class[freq > 0]) # Weight by class frequency

    return {
        'Pixel Accuracy': pixel_accuracy,
        'Mean IoU (mIoU)': m_iou,
        'FWIoU': fwiou,
        'IoU per Class': iou_per_class.tolist(),
        'Confusion Matrix': confusion_matrix.tolist()
    }
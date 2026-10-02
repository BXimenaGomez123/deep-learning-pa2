import torch
import torch.nn.functional as F
import numpy as np
import cv2


def is_binary(mask):
    return torch.all((mask == 0) | (mask == 1))


def iou_score(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape, 'tensors must have the same shape'
    assert is_binary(true_mask), 'tensors must contain only 0/1s'
    assert is_binary(pred_mask), 'tensors must contain only 0/1s'

    intersection = (true_mask * pred_mask).sum()
    union = ((true_mask + pred_mask) > 0).sum()
    return intersection / union


def dice_score(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape, 'tensors must have the same shape'
    assert is_binary(true_mask), 'tensors must contain only 0/1s'
    assert is_binary(pred_mask), 'tensors must contain only 0/1s'

    intersection = (true_mask * pred_mask).sum()
    total = (true_mask + pred_mask).sum()
    return 2 * intersection / total

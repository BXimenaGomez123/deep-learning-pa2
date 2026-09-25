import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple


Box = tuple[int, int, int, int]


def box_iou(box1: Box, box2: Box):
    """IoU between two boxes in (x, y, w, h) format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def match_frame(
        gt_boxes: list[Box],
        gt_ids: list[int],
        pred_boxes: list[Box],
        pred_ids: list[int],
        iou_thresh=0.5) -> list[tuple[int, int]]:
    """Hungarian IoU matching within a single frame -> list of (gt_id, pred_id)."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []

    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_mat[i, j] = box_iou(gb, pb)

    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    return [(int(gt_ids[r]), int(pred_ids[c]))
            for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= iou_thresh]


def count_id_switches(
        gt_boxes: list[list[Box]],
        gt_ids: list[list[int]],
        pred_boxes: list[list[Box]],
        pred_ids: list[list[int]],
        iou_thresh=0.5):
    """Number of times a GT track's matched predicted ID changes frame-to-frame."""
    last_pred_for_gt = {}
    switches = 0
    for t in range(len(gt_boxes)):
        for gt_id, pred_id in match_frame(gt_boxes[t], gt_ids[t], pred_boxes[t], pred_ids[t], iou_thresh):
            if gt_id in last_pred_for_gt and last_pred_for_gt[gt_id] != pred_id:
                switches += 1
            last_pred_for_gt[gt_id] = pred_id
    return switches


def compute_idf1(
        gt_boxes: list[list[Box]],
        gt_ids: list[list[int]],
        pred_boxes: list[list[Box]],
        pred_ids: list[list[int]],
        iou_thresh=0.5):
    """
    Global (not per-frame) identity assignment: for each (gt_id, pred_id) pair,
    count frames where they're IoU-matched; pick the one-to-one gt<->pred
    mapping (Hungarian) maximizing total overlap -> IDTP.
        IDFN = total GT boxes - IDTP,  IDFP = total pred boxes - IDTP
        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
    """
    gt_id_set = sorted(set(int(i) for f in gt_ids for i in f))
    pred_id_set = sorted(set(int(i) for f in pred_ids for i in f))
    gt_index = {gid: i for i, gid in enumerate(gt_id_set)}
    pred_index = {pid: j for j, pid in enumerate(pred_id_set)}

    overlap = np.zeros((len(gt_id_set), len(pred_id_set)))
    for t in range(len(gt_boxes)):
        for gt_id, pred_id in match_frame(gt_boxes[t], gt_ids[t], pred_boxes[t], pred_ids[t], iou_thresh):
            overlap[gt_index[gt_id], pred_index[pred_id]] += 1

    total_gt = sum(len(f) for f in gt_ids)
    total_pred = sum(len(f) for f in pred_ids)

    idtp = 0.0
    if overlap.size > 0:
        row_ind, col_ind = linear_sum_assignment(-overlap)
        idtp = overlap[row_ind, col_ind].sum()

    idfn = total_gt - idtp
    idfp = total_pred - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denom if denom > 0 else 1.0
    return idf1, {"IDTP": idtp, "IDFP": idfp, "IDFN": idfn}

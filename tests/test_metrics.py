import numpy as np
from pa2.metrics import compute_idf1, count_id_switches


def make_track(box, id_val, n_frames):
    boxes = [np.array([box], dtype=np.float32) for _ in range(n_frames)]
    ids = [[id_val] for _ in range(n_frames)]
    return boxes, ids


def merge_tracks(*tracks):
    n_frames = len(tracks[0][0])
    boxes = [np.concatenate([tr[0][t] for tr in tracks]) for t in range(n_frames)]
    ids = [sum((tr[1][t] for tr in tracks), []) for t in range(n_frames)]
    return boxes, ids


def test_perfect_match():
    a = make_track([10, 10, 5, 5], 1, 10)
    b = make_track([50, 50, 5, 5], 2, 10)
    boxes_gt, ids_gt = merge_tracks(a, b)

    idf1, _ = compute_idf1(boxes_gt, ids_gt, boxes_gt, ids_gt)
    switches = count_id_switches(boxes_gt, ids_gt, boxes_gt, ids_gt)

    assert np.isclose(idf1, 1.0)
    assert switches == 0


def test_identity_swap():
    n_frames, k = 10, 5
    a = make_track([10, 10, 5, 5], 1, n_frames)
    b = make_track([50, 50, 5, 5], 2, n_frames)
    boxes_gt, ids_gt = merge_tracks(a, b)

    # same boxes, but the two predicted ids swap from frame k onward
    ids_pred = [ids.copy() for ids in ids_gt]
    for t in range(k, n_frames):
        ids_pred[t] = [2, 1]
    boxes_pred = boxes_gt

    switches = count_id_switches(boxes_gt, ids_gt, boxes_pred, ids_pred)
    idf1, _ = compute_idf1(boxes_gt, ids_gt, boxes_pred, ids_pred)

    assert switches == 2          # cada track sofre exatamente 1 troca
    assert np.isclose(idf1, 0.5)  # IDTP=10, IDFP=IDFN=10 -> 20/40


def test_track_fragmentation():
    n_frames, k = 10, 3  # off-center split so IDTP != 50%
    boxes_gt, ids_gt = make_track([10, 10, 5, 5], 1, n_frames)

    ids_pred = [[1 if t < k else 99] for t in range(n_frames)]
    boxes_pred = boxes_gt

    switches = count_id_switches(boxes_gt, ids_gt, boxes_pred, ids_pred)
    idf1, _ = compute_idf1(boxes_gt, ids_gt, boxes_pred, ids_pred)

    assert switches == 1              # uma única quebra de identidade
    assert np.isclose(idf1, 0.7)      # IDTP=max(3,7)=7 -> 2*7/(2*7+3+3)=14/20
    assert not np.isclose(idf1, 0.5)  # efeito diferente do caso (b)

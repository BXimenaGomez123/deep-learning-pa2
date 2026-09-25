import numpy as np
import torch
from torch.utils.data import Dataset


# ---------- shared generator ----------

def generate_synthetic_video(n_frames=50, n_objects=8, img_size=128,
                             speed=2.0, occlusion_len=10, seed=None):
    """
    Returns:
        frames: [T, H, W, 3] uint8 array (RGB)
        boxes_per_frame: list of length T, each [N_t, 4] (x, y, w, h) - axis-aligned bbox of the ellipse
        ids_per_frame: list of length T, each [N_t] int ids
    Ellipses are drawn with depth order so occlusion is real, each object
    has a fixed random color and elliptical shape (rx, ry, angle).
    """
    rng = np.random.default_rng(seed)
    H = W = img_size

    pos = rng.uniform(10, img_size - 10, size=(n_objects, 2))
    vel = rng.uniform(-speed, speed, size=(n_objects, 2))

    # elliptical shape params per object
    rx = rng.uniform(4, 10, size=n_objects)
    ry = rng.uniform(4, 10, size=n_objects)
    angle = rng.uniform(0, np.pi, size=n_objects)  # fixed rotation per object

    # distinct, saturated random colors per object
    colors = rng.integers(50, 256, size=(n_objects, 3)).astype(np.uint8)

    depth = rng.permutation(n_objects)  # fixed draw order = depth order

    occluded_obj = rng.integers(0, n_objects)
    occ_start = rng.integers(5, max(6, n_frames - occlusion_len - 5))
    occ_end = occ_start + occlusion_len

    frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    boxes_per_frame, ids_per_frame = [], []

    yy, xx = np.ogrid[:H, :W]
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for t in range(n_frames):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        pos = pos + vel
        for d in (0, 1):
            off = (pos[:, d] < 0) | (pos[:, d] > img_size)
            vel[off, d] *= -1
        pos = np.clip(pos, 0, img_size)

        boxes, ids = [], []
        for oid in depth:
            is_occluded_now = (oid == occluded_obj) and (occ_start <= t < occ_end)
            if is_occluded_now:
                continue

            cx, cy = pos[oid]
            a, b = rx[oid], ry[oid]
            ca, sa = cos_a[oid], sin_a[oid]

            # rotated ellipse mask
            dx = xx - cx
            dy = yy - cy
            x_rot = dx * ca + dy * sa
            y_rot = -dx * sa + dy * ca
            mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0

            img[mask] = colors[oid]

            # axis-aligned bounding box of the rotated ellipse
            half_w = np.sqrt((a * ca) ** 2 + (b * sa) ** 2)
            half_h = np.sqrt((a * sa) ** 2 + (b * ca) ** 2)
            x, y, w, h = cx - half_w, cy - half_h, 2 * half_w, 2 * half_h
            boxes.append([x, y, w, h])
            ids.append(oid)

        noise = rng.normal(0, 8, size=img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        frames[t] = img
        boxes_per_frame.append(np.array(boxes, dtype=np.float32).reshape(-1, 4))
        ids_per_frame.append(np.array(ids, dtype=np.int64))

    return frames, boxes_per_frame, ids_per_frame


# ---------- Parte 1: frame-level dataset ----------

class SyntheticFrameDataset(Dataset):
    """One sample = one frame: (image, boxes, ids)."""

    def __init__(self, n_videos=20, **gen_kwargs):
        self.index = []
        self.videos = []
        for v in range(n_videos):
            frames, boxes, ids = generate_synthetic_video(seed=v, **gen_kwargs)
            self.videos.append((frames, boxes, ids))
            for t in range(len(frames)):
                self.index.append((v, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        v, t = self.index[index]
        frames, boxes, ids = self.videos[v]
        # [H,W,3] uint8 -> [3,H,W] float in [0,1]
        image = torch.from_numpy(frames[t]).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(boxes[t]).float()
        ids = torch.from_numpy(ids[t]).int()
        return image, boxes, ids


# ---------- Parte 2: track-level dataset (unchanged, color-agnostic) ----------

class SyntheticTrackDataset(Dataset):
    """One sample = one track's box sequence over a window of T frames,
    built with sliding-window overlap (stride < T)."""

    def __init__(self, n_videos=20, T=16, stride=4, **gen_kwargs):
        self.T = T
        self.samples = []
        for v in range(n_videos):
            frames, boxes, ids = generate_synthetic_video(seed=v, **gen_kwargs)
            n_frames = len(frames)

            all_ids = sorted(set(int(i) for f in ids for i in f))
            for oid in all_ids:
                track = [None] * n_frames
                for t in range(n_frames):
                    hit = np.where(ids[t] == oid)[0]
                    if len(hit) > 0:
                        track[t] = boxes[t][hit[0]]

                for start in range(0, n_frames - T + 1, stride):
                    window = track[start:start + T]
                    box_seq = np.zeros((T, 4), dtype=np.float32)
                    valid = np.zeros((T,), dtype=np.float32)
                    for i, b in enumerate(window):
                        if b is not None:
                            box_seq[i] = b
                            valid[i] = 1.0
                    if valid.sum() >= 2:
                        self.samples.append((box_seq, valid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        box_seq, valid = self.samples[index]
        return torch.from_numpy(box_seq), torch.from_numpy(valid)

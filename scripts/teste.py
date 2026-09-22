import os
import csv
import numpy as np
from PIL import Image
from pa2.plotting import plot_video
from pa2.synthetic_dataset import SyntheticFrameDataset
import matplotlib.pyplot as plt


def load_sequence(seq_name, data_root="data/MOT17/train", start=0, T=50):
    seq_dir = os.path.join(data_root, seq_name)
    img_dir = os.path.join(seq_dir, "img1")

    img_files = sorted(os.listdir(img_dir))[start:start+T]
    frames = [np.array(Image.open(os.path.join(img_dir, f))) for f in img_files]

    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    gt = np.loadtxt(gt_path, delimiter=",") # colunas: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    boxes_per_frame, ids_per_frame = [], []
    for frame_idx in range(start+1, start+T+1): # MOT17 frames são 1-indexed
        rows = gt[gt[:, 0] == frame_idx]
        boxes_per_frame.append(rows[:, 2:6]) # bb_left, bb_top, bb_width, bb_height
        ids_per_frame.append(rows[:, 1].astype(int))

    return frames, boxes_per_frame, ids_per_frame


def load_synthetic_sequence(n_frames=50, n_objects=8, img_size=128):
    dataset = SyntheticFrameDataset(
        n_videos=1,
        n_frames=n_frames,
        n_objects=n_objects,
        img_size=img_size,
    )
    return dataset.videos[0]


if __name__ == "__main__":
    frames, boxes, ids = load_synthetic_sequence()
    ani = plot_video(frames, boxes)
    plt.show()

    frames, boxes, ids = load_sequence("MOT17-02-FRCNN", start=0, T=50)
    ani = plot_video(frames, boxes)
    plt.show()

"""Quick visual sanity check for synthetic and MOT17 boxes."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from pa2.box_corruption import corrupt_track_boxes
from pa2.mot17_dataset import MOT17FrameDataset
from pa2.plotting import plot_video
from pa2.synthetic_dataset import SyntheticFrameDataset


N_FRAMES = 30


def show_video(frames, boxes, ids, title):
    animation = plot_video(frames, boxes, ids)
    plt.gca().set_title(title)
    plt.show()
    plt.close()


def main():
    synthetic = SyntheticFrameDataset(n_videos=1, n_frames=N_FRAMES)
    frames, boxes, ids = synthetic.videos[0]
    show_video(frames, boxes, ids, "Synthetic ellipses — ground truth")

    # Corrupt one object track so the animation remains easy to compare.
    object_id = next(
        oid
        for oid in sorted({int(i) for frame_ids in ids for i in frame_ids})
        if any(oid not in frame_ids for frame_ids in ids)
    )
    track = torch.zeros((N_FRAMES, 4))
    mask = torch.zeros(N_FRAMES)
    for t, (frame_boxes, frame_ids) in enumerate(zip(boxes, ids)):
        matches = np.flatnonzero(frame_ids == object_id)
        if len(matches):
            track[t] = torch.from_numpy(frame_boxes[matches[0]])
            mask[t] = 1

    corrupted_track, corrupted_mask = corrupt_track_boxes(
        track,
        mask,
        drop_fraction=0.15,
        noise_std=2.0,
        false_positive_count=1,
        image_size=(frames.shape[1], frames.shape[2]),
        generator=torch.Generator().manual_seed(0),
    )
    corrupted_boxes, corrupted_ids = [], []
    false_positive_id = max(int(i) for frame_ids in ids for i in frame_ids) + 1
    for t, (frame_boxes, frame_ids) in enumerate(zip(boxes, ids)):
        keep = frame_ids != object_id
        frame_box_list = list(frame_boxes[keep])
        frame_id_list = list(frame_ids[keep])
        if corrupted_mask[t]:
            frame_box_list.append(corrupted_track[t].numpy())
            frame_id_list.append(object_id if mask[t] else false_positive_id)
        corrupted_boxes.append(np.asarray(frame_box_list, dtype=np.float32).reshape(-1, 4))
        corrupted_ids.append(np.asarray(frame_id_list, dtype=np.int64))

    show_video(frames, corrupted_boxes, corrupted_ids, "Synthetic ellipses — corrupted track")

    mot17 = MOT17FrameDataset(root="data", split="train")
    sequence_index = next(
        i for i, sequence in enumerate(mot17.sequences) if sequence.name == "MOT17-02-SDP"
    )
    indices = [i for i, (seq, _) in enumerate(mot17.index) if seq == sequence_index][:N_FRAMES]
    mot_frames, mot_boxes, mot_ids = [], [], []
    for index in indices:
        image, frame_boxes, frame_ids = mot17[index]
        mot_frames.append((image.permute(1, 2, 0) * 255).byte().numpy())
        mot_boxes.append(frame_boxes.numpy())
        mot_ids.append(frame_ids.numpy())
    show_video(mot_frames, mot_boxes, mot_ids, "MOT17-02-SDP — ground truth")


if __name__ == "__main__":
    main()

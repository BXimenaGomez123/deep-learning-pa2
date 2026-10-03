"""Run pretrained Faster R-CNN and GRU tracking on MOT17-13-SDP."""

import matplotlib.pyplot as plt

from pa2.inference import (
    DEFAULT_MAX_FRAMES,
    DETECTION_SCORE_THRESHOLD,
    HISTORY_LENGTH,
    IOU_THRESHOLD,
    MAX_MISSED_FRAMES,
    detect_people,
    run_mot17_gru,
)


SEQUENCE_NAME = "MOT17-13-SDP"
MAX_FRAMES = DEFAULT_MAX_FRAMES


def main():
    video = run_mot17_gru(SEQUENCE_NAME, max_frames=MAX_FRAMES)
    plt.show()
    return video


if __name__ == "__main__":
    main()

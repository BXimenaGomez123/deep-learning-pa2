import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


def plot_video(frames, boxes_per_frame=None, ids_per_frame=None, interval=50):
    """
    frames: lista/array de imagens [T, H, W, C] ou [T, H, W]
    boxes_per_frame: lista de arrays [N_t, 4] (x, y, w, h) por frame (opcional)
    ids_per_frame: lista de arrays [N_t] com ids correspondentes (opcional, pra colorir)
    """
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="gray" if frames[0].ndim == 2 else None)
    patches_list = []

    def update(t):
        im.set_array(frames[t])
        for p in patches_list:
            p.remove()
        patches_list.clear()
        if boxes_per_frame is not None:
            for i, box in enumerate(boxes_per_frame[t]):
                x, y, w, h = box
                color = plt.cm.tab20(ids_per_frame[t][i] % 20) if ids_per_frame is not None else "red"
                rect = patches.Rectangle((x, y), w, h, linewidth=1.5,
                                          edgecolor=color, facecolor="none")
                ax.add_patch(rect)
                patches_list.append(rect)
        return [im] + patches_list

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    return ani

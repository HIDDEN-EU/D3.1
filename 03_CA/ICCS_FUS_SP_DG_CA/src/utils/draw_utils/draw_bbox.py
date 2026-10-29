import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


def draw_bboxes(gt_boxes, pred_boxes, xlim=None, ylim=None):
    def to_numpy(boxes):
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().numpy()
        return np.asarray(boxes).reshape(-1, 7)

    fig, ax = plt.subplots(figsize=(10, 10))

    for boxes, color in [
        (to_numpy(gt_boxes), "green"),
        (to_numpy(pred_boxes), "red"),
    ]:
        for x, y, z, length, width, height, yaw in boxes:
            # Rectangle centered at the origin, length along local x.
            corners = np.array([
                [ length / 2,  width / 2],
                [ length / 2, -width / 2],
                [-length / 2, -width / 2],
                [-length / 2,  width / 2],
            ])

            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s], [s, c]])

            corners = corners @ rotation.T + [x, y]

            ax.add_patch(Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            ))

            # Heading line from center toward the front.
            ax.plot(
                [x, x + length / 2 * c],
                [y, y + length / 2 * s],
                color=color,
            )

    ax.scatter(0, 0, color="black", marker="x", label="Frame origin")
    ax.autoscale_view()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.legend(handles=[
        Line2D([0], [0], color="green", label="Ground truth"),
        Line2D([0], [0], color="red", label="Prediction"),
    ])

    plt.show()
    return fig, ax
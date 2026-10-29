import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from src.utils.centerpoint_utils import get_decoded_bbox_from_pred

def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def draw_box(ax, x, y, length, width, yaw, color):
    # Length along local x; width along local y.
    corners = np.array([
        [-length / 2, -width / 2],
        [ length / 2, -width / 2],
        [ length / 2,  width / 2],
        [-length / 2,  width / 2],
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    corners = corners @ rotation.T + [x, y]

    ax.add_patch(Polygon(
        corners, closed=True, fill=False,
        edgecolor=color, linewidth=1.5, zorder=3,
    ))
    ax.scatter(x, y, color=color, s=10, zorder=4)


@torch.no_grad()
def viz_boxes(
    predictions, targets, pc, score_threshold,
    metres_per_cell, x_min, y_min,
):
    results = get_decoded_bbox_from_pred(
        predictions,
        score_threshold=score_threshold,
        metres_per_cell=metres_per_cell,
        x_min=x_min,
        y_min=y_min,
    )

    for b, result in enumerate(results):
        boxes = to_numpy(result["boxes"])
        points = to_numpy(pc[b])

        # The repo's head already applies sigmoid.
        heatmap = predictions["heatmap"][b]
        print(
            f"Sample {b}: threshold={score_threshold}, "
            f"max score={heatmap.max().item():.4f}, "
            f"retained boxes={len(boxes)}"
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(
            points[:, 0], points[:, 1],
            c=points[:, 2], cmap="viridis",
            s=0.5, alpha=0.4, zorder=1,
        )

        # Each sample contains a list of GT object dictionaries.
        for obj in targets[b]:
            x, y = [float(v) for v in obj["bbox"]]
            dims = obj["3d_dimensions"]
            draw_box(
                ax, x, y,
                float(dims["l"]), float(dims["w"]),
                float(obj["rotation"]), "green",
            )

        # Each prediction row is [x, y, z, l, w, h, yaw].
        for x, y, z, length, width, height, yaw in boxes:
            draw_box(ax, x, y, length, width, yaw, "red")

        ax.set_title(
            f"Sample {b}: {len(boxes)} predictions, "
            f"{len(targets[b])} targets"
        )
        ax.set(xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale_view()
        ax.grid(alpha=0.3)
        ax.legend(handles=[
            Line2D([0], [0], color="green", label="Target"),
            Line2D([0], [0], color="red", label="Prediction"),
        ])

        plt.show()
        plt.close(fig)
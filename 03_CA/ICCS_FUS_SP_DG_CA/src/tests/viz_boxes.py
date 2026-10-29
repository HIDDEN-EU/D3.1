import os
import sys
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.transforms import Affine2D
from src.utils.centerpoint_utils import get_decoded_bbox_from_pred, nms_boxes


def viz_boxes(predictions ,
              targets,
              pc,
              score_threshold,
              metres_per_cell,
              x_min,
              y_min,
              save_path=None,
              desc=None)->None:
    """
    visualize 2d boxes from prediction and targert
    """
    class_names=["Car","Truck","Van","Bus","Pedestrian","Cyclist","Tricyclist",     "Motorcyclist","Barrowlist","Trafficcone"]


    B = predictions["heatmap"].shape[0]
    results = get_decoded_bbox_from_pred(predictions,
                                         score_threshold=score_threshold,metres_per_cell=metres_per_cell,
                                         x_min=x_min,
                                         y_min=y_min)
    
    results = nms_boxes(results)
    
    print("pc shape:", pc.shape)

    # pc=pc.detach().cpu().tolist()
    # pc=np.asarray(pc)[:,:3]
    # print(pc)
    # sys.exit()

    if save_path is None:
        print("skipping saving plt")

    if save_path is not None:
        fig,ax=plt.subplots()

        for batch in range(B):
            print("batcvh is {}".format(batch))
            pc = pc[batch, :, :3].detach().cpu().numpy()    
            boxes = results[batch]["boxes"].detach().cpu().tolist()

            labels = results[batch]["labels"].detach().cpu().tolist()

            if 1:
                for box,label in zip(boxes,labels):  #multiple box
                    x_p, y_p, z_p , l_p , w_p, h_p, yaw_p = box
                    
                    rectangle=Rectangle(  #prediction
                            ( x_p - w_p / 2,y_p - l_p / 2),
                            l_p,
                            w_p,
                            fill=False,
                            edgecolor="red",
                            linewidth=2,
                            label="predict",
                        )
                    rectangle.set_transform(
                        Affine2D().rotate_around(x_p, y_p, yaw_p) + ax.transData
                    )
                    ax.add_patch(rectangle)

                    # Highest y-coordinate of the rotated rectangle.
                    top_y = (
                        y_p
                        + abs(np.sin(yaw_p)) * w_p / 2
                        + abs(np.cos(yaw_p)) * l_p / 2
                    )

                    ax.annotate(
                        class_names[label],              # Or class_names[label]
                        xy=(x_p, top_y),
                        xytext=(0, 4),           # Offset above the box in points
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="red",
                            edgecolor="none",
                            alpha=0.8,
                        ),)
            
            if targets is not None:
                for gt_box in targets["gt_boxes"][batch]:
                    x, y =  gt_box["bbox"]#targets
                    l,w= gt_box["3d_dimensions"]["l"],gt_box["3d_dimensions"]["w"]
                    yaw = gt_box["rotation"]
                    # print(x,y,l,w)
                    # sys.exit()
                    rectangle = Rectangle(  #target
                                (x - w/2 , y - l/ 2),
                                w,
                                l,
                                fill=False,
                                edgecolor="green",
                                linewidth=2,
                                label="target",
                            )
                    rectangle.set_transform(
                        Affine2D().rotate_around(x, y, yaw) + ax.transData
                    )
                    ax.add_patch(rectangle)

            ax.set_aspect("equal", adjustable="box")

            ax.set_title("object predictions vs targets")
   
            print("XYZ min:", pc[:, :3].min(axis=0))
            print("XYZ max:", pc[:, :3].max(axis=0))
            print("All finite:", np.isfinite(pc[:, :3]).all())
            
        if 1:
            print(f"point cloud sdhape {pc.shape}")
            ax.scatter( pc[:,0] , 
                        pc[:,1], 
                        c="blue",
                        cmap="viridis",
                        s=0.5,
                        alpha=0.5, 
                        zorder=1 )

        ax.set(xlabel="x", 
                ylabel="y")
        
        ax.grid(True)
        
        print("saving plt")
        plt.savefig(os.path.join(save_path + desc+".png"),
                    dpi=300,
                    bbox_inches="tight")
        # plt.savefig(os.path.join(save_path + desc+".svg"),
        #             dpi=300,
        #             bbox_inches="tight")
        plt.close(fig)

        

import sys
from pypcd import pypcd 
import mmcv
import numpy as np
import torch 

def read_pcd(path_pcd):
        """
        read and preprocess poing cloud 
        """
        pcd = pypcd.PointCloud.from_path(path_pcd)
        pcd_np_points = np.zeros(
            (pcd.points, 4),
            dtype=np.float32
        )

        pcd_np_points[:, 0] = pcd.pc_data["x"]
        pcd_np_points[:, 1] = pcd.pc_data["y"]
        pcd_np_points[:, 2] = pcd.pc_data["z"]

        pcd_np_points[:, 3] = (
            pcd.pc_data["intensity"] / 256.0
        )

        del_index = np.where(
            np.isnan(pcd_np_points)
        )[0]

        pcd_np_points = np.delete(
            pcd_np_points,
            del_index,
            axis=0
        )
        return torch.tensor(pcd_np_points)

def read_jpg(jpg_path):
    image = mmcv.imread(jpg_path)
    # print("read image dtpe , shape{},{}".format(type(image.dtype),image.shape))
    # sys.exit()
    return image
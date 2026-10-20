import numpy as np


def compose_ego2img(ego2global_t,
                    ego2global_r,
                    sensor2global_t,
                    sensor2global_r,
                    cam_intrinsic):
    R = np.linalg.inv(sensor2global_r) @ ego2global_r
    # (ego2global_t - sensor2global_t) @ _inv(sensor2global_r).T
    # = (ego2global_t - sensor2global_t) @ sensor2global_r
    T = (ego2global_t - sensor2global_t) @ sensor2global_r

    ego2cam_rt = np.eye(4)
    ego2cam_rt[:3, :3] = R
    ego2cam_rt[:3, 3] = T.T

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
    ego2img = (viewpad @ ego2cam_rt).astype(np.float32)

    return ego2img
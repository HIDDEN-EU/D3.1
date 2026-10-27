
import numpy as np
import matplotlib.pyplot as plt

img = np.load("/mnt/e/dev/hidden/projects/detection_dairv2x/extracted/recording_0000/_my_camera_pylon_ros2_camera_node_image_raw_72d651f84e42/000001768_1791209805285588641.npy", allow_pickle=False)
print("Shape:", img.shape, "dtype:", img.dtype)

# If the original ROS image encoding was "bgr8", uncomment:
# img = img[..., ::-1]

print(img.shape)

plt.figure(figsize=(12, 7))
plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
plt.axis("off")
plt.show()

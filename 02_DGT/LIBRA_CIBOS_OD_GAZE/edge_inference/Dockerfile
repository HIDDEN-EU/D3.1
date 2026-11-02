# syntax=docker/dockerfile:1.7
#
# The one DGT image: dgt-daemon, the engine builder and rosbridge all run from
# it (compose.yaml picks the command). Jetson only — arm64 + L4T.
#
# Build (on the Jetson, from this folder; the context is the parent folder so
# the sibling road_object_detection package is available):
#   docker compose build
#
# Base: NVIDIA's official JetPack 6 image, with CUDA 12.6, TensorRT 10.3, and
# OpenCV with GStreamer. nvarguscamerasrc comes in through the nvidia runtime.
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ros2-apt-source: ROS's own way to install the apt key and source. Pinned so a
# rebuild resolves the same repo.
ARG ROS_APT_SOURCE_VERSION=1.3.0

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    ROS_DOMAIN_ID=0 \
    ROS_LOCALHOST_ONLY=1

# Never `apt-get upgrade` here: it would pull L4T / TensorRT packages out of
# step with the base image, and an engine is valid only for the TensorRT that
# built it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl python3-pip python3-venv \
 && curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb" \
 && apt-get install -y --no-install-recommends /tmp/ros2-apt-source.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends ros-humble-ros-base ros-humble-rosbridge-suite \
 && rm -rf /var/lib/apt/lists/* /tmp/ros2-apt-source.deb

# --system-site-packages keeps the image's cv2 (GStreamer) and TensorRT visible;
# pip is upgraded first because the stock 22.0.2 builds packages as UNKNOWN-0.0.0.
RUN python3 -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /opt/dgt

# The road-object tiling recipe first, so an edge source change does not
# reinstall it. Base install only (NumPy).
COPY road_object_detection/ road_object_detection/
RUN pip install --no-cache-dir ./road_object_detection

# A wheel install, never `-e .`.
COPY edge_inference/pyproject.toml edge_inference/README.md edge_inference/
COPY edge_inference/dgt/ edge_inference/dgt/
COPY edge_inference/scripts/ edge_inference/scripts/
RUN pip install --no-cache-dir './edge_inference[jetson,daemon]' \
 && command -v dgt-daemon dgt-build-engines

COPY edge_inference/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["dgt-daemon"]

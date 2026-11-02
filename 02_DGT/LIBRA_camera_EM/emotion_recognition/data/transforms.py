"""
Augmentation pipelines for FER2013 (grayscale) and KDEF (RGB).

Statistics
----------
FER2013  – computed from the training set (48×48 grayscale).
           Images are opened as RGB (3 identical channels), so the
           same scalar is replicated across all three channels.
           mean=(0.5071, 0.5071, 0.5071)  std=(0.2212, 0.2212, 0.2212)

KDEF     – ImageNet RGB statistics used as initialisation prior;
           replace with dataset-specific values after running
           `compute_dataset_stats` on the actual training split.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torchvision import transforms

if TYPE_CHECKING:
    from emotion_recognition.config import AugmentationConfig

# ── Per-dataset normalisation constants ───────────────────────────────────────

# FER2013 images are grayscale stored as RGB — same value on all 3 channels.
FER2013_MEAN = (0.5071, 0.5071, 0.5071)
FER2013_STD  = (0.2212, 0.2212, 0.2212)

KDEF_MEAN  = (0.485, 0.456, 0.406)   # ImageNet prior
KDEF_STD   = (0.229, 0.224, 0.225)

# RAF-DB / AffectNet: web-collected RGB — ImageNet stats are a good prior
RAFDB_MEAN     = (0.485, 0.456, 0.406)
RAFDB_STD      = (0.229, 0.224, 0.225)
AFFECTNET_MEAN = RAFDB_MEAN
AFFECTNET_STD  = RAFDB_STD

# FER+ images are extracted from the same FER2013 pixel CSV — identical stats
FERPLUS_MEAN = FER2013_MEAN
FERPLUS_STD  = FER2013_STD

# SwinFace face-recognition normalisation: pixel values mapped to [-1, 1]
SWINFACE_MEAN = (0.5, 0.5, 0.5)
SWINFACE_STD  = (0.5, 0.5, 0.5)

# ── Input sizes ───────────────────────────────────────────────────────────────

FER2013_SIZE  = 48    # native resolution — most architectures upsample internally
KDEF_SIZE     = 224   # standard CNN input; KDEF native is 562×762
RAFDB_SIZE    = 224   # RAF-DB images are variable size; aligned crops ~100px
FERPLUS_SIZE  = 48    # same as FER2013 — extracted from the same pixel CSV
SWINFACE_SIZE = 112   # SwinFace backbone expects 112×112 (face-recognition convention)


def fer2013_transforms(split: str, aug: "AugmentationConfig | None" = None) -> transforms.Compose:
    """
    Returns a transform pipeline for FER2013.

    Parameters
    ----------
    split : {'train', 'test'}
    aug : AugmentationConfig, optional
        Augmentation parameters. ``None`` values fall back to dataset defaults.
    """
    from emotion_recognition.config import AugmentationConfig
    if aug is None:
        aug = AugmentationConfig()

    rotation   = aug.rotation_degrees if aug.rotation_degrees is not None else 10
    scale_min  = aug.crop_scale_min   if aug.crop_scale_min   is not None else 0.85

    if split == "train":
        pipeline = []
        if aug.horizontal_flip:
            pipeline.append(transforms.RandomHorizontalFlip())
        pipeline += [
            transforms.RandomRotation(degrees=rotation),
            transforms.RandomResizedCrop(
                size=FER2013_SIZE,
                scale=(scale_min, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomAutocontrast(p=aug.autocontrast_p),
            transforms.ToTensor(),
            transforms.Normalize(FER2013_MEAN, FER2013_STD),
        ]
        return transforms.Compose(pipeline)
    else:
        return transforms.Compose([
            transforms.Resize(FER2013_SIZE,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(FER2013_MEAN, FER2013_STD),
        ])


def kdef_transforms(split: str, aug: "AugmentationConfig | None" = None) -> transforms.Compose:
    """
    Returns a transform pipeline for KDEF.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
    aug : AugmentationConfig, optional
        Augmentation parameters. ``None`` values fall back to dataset defaults.
    """
    from emotion_recognition.config import AugmentationConfig
    if aug is None:
        aug = AugmentationConfig()

    rotation   = aug.rotation_degrees        if aug.rotation_degrees        is not None else 12
    scale_min  = aug.crop_scale_min          if aug.crop_scale_min          is not None else 0.80
    brightness = aug.color_jitter_brightness if aug.color_jitter_brightness is not None else 0.3
    contrast   = aug.color_jitter_contrast   if aug.color_jitter_contrast   is not None else 0.3
    saturation = aug.color_jitter_saturation if aug.color_jitter_saturation is not None else 0.2
    erasing_p  = aug.random_erasing_p        if aug.random_erasing_p        is not None else 0.25

    if split == "train":
        pipeline: list = [
            transforms.RandomResizedCrop(
                size=KDEF_SIZE,
                scale=(scale_min, 1.0),
                ratio=(0.90, 1.10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
        if aug.horizontal_flip:
            pipeline.append(transforms.RandomHorizontalFlip())
        pipeline += [
            transforms.RandomRotation(degrees=rotation),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=aug.color_jitter_hue,
            ),
            transforms.RandomGrayscale(p=aug.random_grayscale_p),
            transforms.ToTensor(),
            transforms.Normalize(KDEF_MEAN, KDEF_STD),
            # Random erasing simulates occlusion (glasses, hands, mask)
            transforms.RandomErasing(
                p=erasing_p,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
        return transforms.Compose(pipeline)
    else:
        return transforms.Compose([
            transforms.Resize(
                int(KDEF_SIZE * 256 / 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(KDEF_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(KDEF_MEAN, KDEF_STD),
        ])


def ferplus_transforms(split: str, aug: "AugmentationConfig | None" = None) -> transforms.Compose:
    """
    Returns a transform pipeline for FER+.

    FER+ images are 48×48 grayscale PNGs extracted from the original FER2013
    pixel CSV, so the statistics and augmentation strategy match FER2013.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
    aug : AugmentationConfig, optional
        Augmentation parameters. ``None`` values fall back to dataset defaults.
    """
    return fer2013_transforms(split, aug)


def swinface_transforms(
    split: str,
    dataset: str,
    aug: "AugmentationConfig | None" = None,
) -> transforms.Compose:
    """
    Returns a transform pipeline for any dataset when using the SwinFace backbone.

    The SwinFace backbone expects 112×112 RGB inputs normalised to [-1, 1].
    For grayscale datasets (FER2013, FER+) a ``Grayscale(3)`` step replicates
    the single channel to 3 channels before resizing — no learned projection needed.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
    dataset : str
        One of ``'fer2013'``, ``'ferplus'``, ``'kdef'``, ``'rafdb'``.
    aug : AugmentationConfig, optional
        Augmentation parameters. ``None`` values fall back to dataset defaults.
    """
    from emotion_recognition.config import AugmentationConfig
    if aug is None:
        aug = AugmentationConfig()

    is_grayscale = dataset in ("fer2013", "ferplus")

    rotation  = aug.rotation_degrees if aug.rotation_degrees is not None else 10
    scale_min = aug.crop_scale_min   if aug.crop_scale_min   is not None else 0.85

    if split == "train":
        pipeline: list = []
        if is_grayscale:
            pipeline.append(transforms.Grayscale(num_output_channels=3))
        if aug.horizontal_flip:
            pipeline.append(transforms.RandomHorizontalFlip())
        pipeline += [
            transforms.RandomRotation(degrees=rotation),
            transforms.RandomResizedCrop(
                size=SWINFACE_SIZE,
                scale=(scale_min, 1.0),
                ratio=(0.90, 1.10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(SWINFACE_MEAN, SWINFACE_STD),
        ]
        return transforms.Compose(pipeline)
    else:
        pipeline = []
        if is_grayscale:
            pipeline.append(transforms.Grayscale(num_output_channels=3))
        pipeline += [
            transforms.Resize(
                SWINFACE_SIZE,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(SWINFACE_MEAN, SWINFACE_STD),
        ]
        return transforms.Compose(pipeline)


def affectnet_transforms(split: str, aug: "AugmentationConfig | None" = None) -> transforms.Compose:
    """AffectNet is RGB in-the-wild — same pipeline as RAF-DB."""
    return rafdb_transforms(split, aug)


def rafdb_transforms(split: str, aug: "AugmentationConfig | None" = None) -> transforms.Compose:
    """
    Returns a transform pipeline for RAF-DB.

    RAF-DB images are real-world, already face-aligned, RGB.
    Stronger augmentation than KDEF to counter the in-the-wild noise.

    Parameters
    ----------
    split : {'train', 'val', 'test'}
    aug : AugmentationConfig, optional
        Augmentation parameters. ``None`` values fall back to dataset defaults.
    """
    from emotion_recognition.config import AugmentationConfig
    if aug is None:
        aug = AugmentationConfig()

    rotation   = aug.rotation_degrees        if aug.rotation_degrees        is not None else 15
    scale_min  = aug.crop_scale_min          if aug.crop_scale_min          is not None else 0.75
    brightness = aug.color_jitter_brightness if aug.color_jitter_brightness is not None else 0.4
    contrast   = aug.color_jitter_contrast   if aug.color_jitter_contrast   is not None else 0.4
    saturation = aug.color_jitter_saturation if aug.color_jitter_saturation is not None else 0.3
    erasing_p  = aug.random_erasing_p        if aug.random_erasing_p        is not None else 0.30

    if split == "train":
        pipeline: list = [
            transforms.RandomResizedCrop(
                size=RAFDB_SIZE,
                scale=(scale_min, 1.0),
                ratio=(0.90, 1.10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
        if aug.horizontal_flip:
            pipeline.append(transforms.RandomHorizontalFlip())
        pipeline += [
            transforms.RandomRotation(degrees=rotation),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=aug.color_jitter_hue,
            ),
            transforms.RandomGrayscale(p=aug.random_grayscale_p),
            transforms.ToTensor(),
            transforms.Normalize(RAFDB_MEAN, RAFDB_STD),
            transforms.RandomErasing(
                p=erasing_p,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
        return transforms.Compose(pipeline)
    else:
        return transforms.Compose([
            transforms.Resize(
                int(RAFDB_SIZE * 256 / 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(RAFDB_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(RAFDB_MEAN, RAFDB_STD),
        ])

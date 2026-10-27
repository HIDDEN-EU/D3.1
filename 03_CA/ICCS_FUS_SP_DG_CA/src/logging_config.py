import logging
from pathlib import Path


def setup_logging(
    log_file="/mnt/e/dev/hidden/projects/detection_dairv2x/logging/train.log",
    level=logging.INFO,
):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(pathname)s:%(lineno)d | %(message)s",
        handlers=[
            # logging.StreamHandler(),
            logging.FileHandler(log_path , mode="w")
        ],
        force=True,
    )
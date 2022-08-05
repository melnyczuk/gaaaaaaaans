from os import mkdir, path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


def save_images(
    imgs: Iterable[np.ndarray],
    dir: str = "./output/cloudput",
    name: Optional[str] = None,
) -> None:
    if not path.isdir(dir):
        mkdir(dir)
    for idx, img in enumerate(imgs):
        (
            Image.fromarray(((img + 1) * 127.5).astype(np.uint8))
            .convert("RGB")
            .save(f"{dir}/{name}::{idx}.jpeg")
        )

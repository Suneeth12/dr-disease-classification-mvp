from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from retina_api.core.settings import IMAGE_SIZE


def preprocess_fundus_for_inference(image_path: str | Path, img_size: int = IMAGE_SIZE) -> dict[str, np.ndarray | str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.resize(original_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    lab_image = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_lightness = clahe.apply(lightness)
    processed_lab = cv2.merge((equalized_lightness, a_channel, b_channel))
    processed_rgb = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(processed_rgb, axis=0).astype(np.float32)

    return {
        "image_path": str(image_path),
        "original_rgb": original_rgb,
        "processed_rgb": processed_rgb,
        "img_tensor": img_tensor,
    }

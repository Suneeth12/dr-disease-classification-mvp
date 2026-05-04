from pathlib import Path

import cv2
import numpy as np

from retina_api.ml.preprocessing import preprocess_fundus_for_inference


def test_preprocess_fundus_for_inference_returns_notebook_contract(tmp_path: Path) -> None:
    image_path = tmp_path / "fundus.png"
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.circle(image, (160, 160), 120, (45, 120, 210), thickness=-1)
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    result = preprocess_fundus_for_inference(image_path)

    assert result["image_path"] == str(image_path)
    assert result["original_rgb"].shape == (224, 224, 3)
    assert result["processed_rgb"].shape == (224, 224, 3)
    assert result["img_tensor"].shape == (1, 224, 224, 3)
    assert result["processed_rgb"].dtype == np.float32
    assert result["img_tensor"].dtype == np.float32
    assert float(result["processed_rgb"].min()) >= 0.0
    assert float(result["processed_rgb"].max()) <= 1.0

import numpy as np

from retina_api.ml.patch_mil import build_patch_mil_patch_bag, build_retina_mask


def test_build_retina_mask_detects_foreground_region() -> None:
    image = np.zeros((224, 224, 3), dtype=np.float32)
    yy, xx = np.ogrid[:224, :224]
    mask = (xx - 112) ** 2 + (yy - 112) ** 2 <= 85**2
    image[mask] = np.array([0.25, 0.55, 0.85], dtype=np.float32)

    retina_mask = build_retina_mask(image)

    assert retina_mask.shape == (224, 224)
    assert retina_mask.dtype == np.float32
    assert float(retina_mask.max()) == 1.0
    assert float(retina_mask.mean()) > 0.0


def test_build_patch_mil_patch_bag_returns_fixed_size_payload() -> None:
    image = np.zeros((224, 224, 3), dtype=np.float32)
    yy, xx = np.ogrid[:224, :224]
    mask = (xx - 112) ** 2 + (yy - 112) ** 2 <= 92**2
    image[mask] = np.array([0.35, 0.60, 0.90], dtype=np.float32)

    patch_bag = build_patch_mil_patch_bag(image)

    assert patch_bag["patch_images"].shape == (9, 224, 224, 3)
    assert patch_bag["patch_boxes"].shape == (9, 4)
    assert patch_bag["patch_valid_mask"].shape == (9,)
    assert patch_bag["patch_valid_mask"].dtype == np.float32
    assert patch_bag["patch_count_after_padding"] == 9
    assert patch_bag["patch_count_before_padding"] > 0

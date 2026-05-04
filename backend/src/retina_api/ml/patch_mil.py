from __future__ import annotations

import cv2
import numpy as np


def build_retina_mask(image_rgb: np.ndarray) -> np.ndarray:
    image_uint8 = np.clip(np.asarray(image_rgb) * 255.0, 0.0, 255.0).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    binary_mask = (gray > 10).astype(np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled_mask, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
    else:
        filled_mask = binary_mask
    blurred_mask = cv2.GaussianBlur(filled_mask.astype(np.float32), (11, 11), 0)
    if float(np.max(blurred_mask)) > 0.0:
        blurred_mask = blurred_mask / float(np.max(blurred_mask))
    return (blurred_mask >= 0.2).astype(np.float32)


def build_patch_mil_patch_bag(processed_rgb: np.ndarray) -> dict[str, np.ndarray | int]:
    patch_size = 112
    stride = 56
    max_patches = 9
    resize_target_shape = (224, 224)

    processed_rgb = np.asarray(processed_rgb, dtype=np.float32)
    if processed_rgb.shape[:2] != resize_target_shape:
        raise ValueError("Patch-MIL helper expects a preprocessed 224x224 RGB image")

    retina_mask = build_retina_mask(processed_rgb)
    retained_patch_images = []
    retained_patch_boxes = []
    retained_patch_coverage = []
    image_height, image_width = processed_rgb.shape[:2]

    for y0 in range(0, image_height - patch_size + 1, stride):
        for x0 in range(0, image_width - patch_size + 1, stride):
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            patch_mask = retina_mask[y0:y1, x0:x1]
            retinal_coverage = float(np.mean(patch_mask)) if patch_mask.size else 0.0
            if retinal_coverage >= 0.35:
                patch_image = processed_rgb[y0:y1, x0:x1]
                patch_image = cv2.resize(
                    patch_image,
                    resize_target_shape,
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float32)
                retained_patch_images.append(patch_image)
                retained_patch_boxes.append((x0, y0, x1, y1))
                retained_patch_coverage.append(retinal_coverage)

    patch_count_before_padding = min(len(retained_patch_images), max_patches)
    patch_images = np.zeros((max_patches, 224, 224, 3), dtype=np.float32)
    patch_boxes = np.zeros((max_patches, 4), dtype=np.float32)
    patch_valid_mask = np.zeros((max_patches,), dtype=np.float32)
    patch_retinal_coverage = np.zeros((max_patches,), dtype=np.float32)

    for index in range(patch_count_before_padding):
        patch_images[index] = retained_patch_images[index]
        patch_boxes[index] = np.asarray(retained_patch_boxes[index], dtype=np.float32)
        patch_valid_mask[index] = 1.0
        patch_retinal_coverage[index] = float(retained_patch_coverage[index])

    return {
        "patch_images": patch_images,
        "patch_boxes": patch_boxes,
        "patch_valid_mask": patch_valid_mask,
        "patch_retinal_coverage": patch_retinal_coverage,
        "patch_count_before_padding": patch_count_before_padding,
        "patch_count_after_padding": int(max_patches),
    }


def build_patch_mil_single_image_inputs(processed_rgb: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | int]]:
    patch_bag = build_patch_mil_patch_bag(processed_rgb)
    patch_inputs = {
        "patch_images": patch_bag["patch_images"][np.newaxis, ...].astype(np.float32),
        "patch_boxes": patch_bag["patch_boxes"][np.newaxis, ...].astype(np.float32),
        "patch_valid_mask": patch_bag["patch_valid_mask"][np.newaxis, ...].astype(np.float32),
    }
    return patch_inputs, patch_bag

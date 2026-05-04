from __future__ import annotations

import numpy as np


def expected_grade_from_probs(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError("Expected probabilities with shape [n_samples, n_classes]")
    grade_axis = np.arange(probabilities.shape[1], dtype=np.float32)
    return probabilities @ grade_axis


def apply_thresholds(expected_grades: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    expected_grades = np.asarray(expected_grades, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (4,):
        raise ValueError("Expected exactly four thresholds for five DR grades")
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("Thresholds must be strictly increasing")
    return np.digitize(expected_grades, thresholds, right=False).astype(np.int32)


def logits_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-7, 1.0)
    return np.log(probabilities)

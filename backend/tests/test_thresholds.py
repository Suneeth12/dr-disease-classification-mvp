import numpy as np
import pytest

from retina_api.ml.thresholds import apply_thresholds, expected_grade_from_probs


def test_expected_grade_from_probs_returns_weighted_class_average() -> None:
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.25, 0.50],
        ],
        dtype=np.float32,
    )

    grades = expected_grade_from_probs(probabilities)

    assert np.allclose(grades, np.array([0.40, 3.25], dtype=np.float32))


def test_apply_thresholds_maps_expected_grades_to_five_classes() -> None:
    expected_grades = np.array([0.20, 0.80, 1.70, 2.70, 3.20], dtype=np.float32)
    thresholds = np.array([0.35, 1.25, 2.25, 3.05], dtype=np.float32)

    predictions = apply_thresholds(expected_grades, thresholds)

    assert predictions.tolist() == [0, 1, 2, 3, 4]


def test_apply_thresholds_rejects_non_monotonic_vectors() -> None:
    expected_grades = np.array([1.2], dtype=np.float32)

    with pytest.raises(ValueError, match="strictly increasing"):
        apply_thresholds(expected_grades, np.array([0.35, 1.25, 1.25, 3.05], dtype=np.float32))

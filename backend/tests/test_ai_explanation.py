from retina_api.ml.explanation import build_ai_explanation
from retina_api.ml.ereg_graph import build_ereg_graph


def test_build_ai_explanation_uses_multiscale_heatmap_and_simple_confidence_text() -> None:
    artifacts = [
        {
            "model_name": "multiscale",
            "display_name": "Multiscale",
            "kind": "gradcam",
            "focus_region": "upper-central retina",
            "focus_compactness": "localized",
            "focus_center": [0.53, 0.26],
            "focus_coverage": 0.07,
        },
    ]

    explanation = build_ai_explanation(
        predicted_class_index=2,
        predicted_label="Moderate",
        confidence=0.74,
        class_probabilities=[0.04, 0.12, 0.74, 0.06, 0.04],
        gradcam_artifacts=artifacts,
    )

    assert explanation["focus_region"] == "upper-central retina"
    assert explanation["model_agreement"] == "single Grad-CAM"
    assert explanation["supporting_models"] == []
    assert explanation["class_margin"] == 0.62
    assert "upper-central retina" in explanation["summary"]
    assert "Moderate" in explanation["summary"]
    combined_text = " ".join(
        [
            explanation["summary"],
            explanation["confidence_reason"],
            explanation["limitations"],
        ]
    ).lower()
    assert "supporting model" not in combined_text
    assert "multiscale" not in combined_text
    assert "patch" not in combined_text
    assert "attention" not in combined_text


def test_build_ereg_graph_uses_dynamic_weights_and_gradcam_focus() -> None:
    graph = build_ereg_graph(
        predicted_class_index=2,
        predicted_label="Moderate",
        confidence=0.74,
        expected_grade=1.95,
        class_probabilities=[0.04, 0.12, 0.74, 0.06, 0.04],
        ensemble_members=["attention", "patch_mil", "lesion", "multiscale"],
        ensemble_member_weights=[0.25, 0.20, 0.30, 0.25],
        threshold_vector=[0.69, 1.17, 2.83, 3.55],
        gradcam_artifacts=[
            {
                "model_name": "multiscale",
                "focus_region": "upper-central retina",
                "focus_compactness": "localized",
            }
        ],
    )

    member_nodes = [node for node in graph["nodes"] if node["kind"] == "member"]
    assert [node["id"] for node in member_nodes] == ["attention", "patch_mil", "lesion", "multiscale"]
    assert [node["weight"] for node in member_nodes] == [0.25, 0.2, 0.3, 0.25]
    threshold_node = next(node for node in graph["nodes"] if node["id"] == "ordinal_thresholds")
    assert threshold_node["detail"] == "Severity score 1.95 falls in the Moderate range."
    visual_node = next(node for node in graph["nodes"] if node["id"] == "gradcam")
    assert visual_node["label"] == "Grad-CAM"
    assert "E-REG++" not in graph["summary"]
    assert "upper-central retina" in graph["summary"]

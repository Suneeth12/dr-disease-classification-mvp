import csv
from pathlib import Path

from retina_api.ml.sample_data import materialize_sample_dataset, select_balanced_rows


def write_train_fixture(root: Path) -> tuple[Path, Path]:
    csv_path = root / "train.csv"
    images_dir = root / "train_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"id_code": "0001", "diagnosis": "0"},
        {"id_code": "0002", "diagnosis": "0"},
        {"id_code": "0101", "diagnosis": "1"},
        {"id_code": "0102", "diagnosis": "1"},
        {"id_code": "0201", "diagnosis": "2"},
        {"id_code": "0202", "diagnosis": "2"},
        {"id_code": "0301", "diagnosis": "3"},
        {"id_code": "0302", "diagnosis": "3"},
        {"id_code": "0401", "diagnosis": "4"},
        {"id_code": "0402", "diagnosis": "4"},
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        (images_dir / f"{row['id_code']}.png").write_bytes(b"png-bytes")

    return csv_path, images_dir


def test_select_balanced_rows_returns_first_two_rows_per_class(tmp_path: Path) -> None:
    csv_path, _ = write_train_fixture(tmp_path)

    rows = select_balanced_rows(csv_path, per_class=2)

    assert [row["id_code"] for row in rows] == [
        "0001",
        "0002",
        "0101",
        "0102",
        "0201",
        "0202",
        "0301",
        "0302",
        "0401",
        "0402",
    ]


def test_materialize_sample_dataset_creates_csv_and_image_folder(tmp_path: Path) -> None:
    csv_path, images_dir = write_train_fixture(tmp_path / "source")
    output_dir = tmp_path / "sample_output"

    dataset = materialize_sample_dataset(
        train_csv_path=csv_path,
        train_images_dir=images_dir,
        output_dir=output_dir,
        per_class=2,
    )

    assert dataset.csv_path.exists()
    assert dataset.images_dir.exists()
    assert (dataset.images_dir / "0001.png").exists()
    assert (output_dir / "README.txt").exists()

    with dataset.csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 10
    assert rows[0]["image_path"] == "images/0001.png"

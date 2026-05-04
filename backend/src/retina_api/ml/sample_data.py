from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


SAMPLE_COLUMNS = ["id_code", "diagnosis", "image_filename", "image_path"]


@dataclass(frozen=True)
class SampleDataset:
    output_dir: Path
    csv_path: Path
    images_dir: Path
    rows: list[dict[str, str]]


def _read_train_rows(train_csv_path: Path) -> list[dict[str, str]]:
    with train_csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_balanced_rows(
    train_csv_path: Path,
    *,
    per_class: int = 2,
) -> list[dict[str, str]]:
    rows = _read_train_rows(train_csv_path)
    grouped: dict[str, list[dict[str, str]]] = {}

    for row in sorted(rows, key=lambda item: (int(item["diagnosis"]), item["id_code"])):
        grouped.setdefault(row["diagnosis"], []).append(row)

    selected_rows: list[dict[str, str]] = []
    for diagnosis in sorted(grouped.keys(), key=int):
        class_rows = grouped[diagnosis]
        if len(class_rows) < per_class:
            raise ValueError(
                f"Diagnosis class {diagnosis} only has {len(class_rows)} rows; "
                f"cannot select {per_class} samples."
            )
        selected_rows.extend(class_rows[:per_class])

    return selected_rows


def materialize_sample_dataset(
    *,
    train_csv_path: Path,
    train_images_dir: Path,
    output_dir: Path,
    per_class: int = 2,
) -> SampleDataset:
    selected_rows = select_balanced_rows(train_csv_path, per_class=per_class)
    images_dir = output_dir / "images"
    csv_path = output_dir / "sample.csv"
    readme_path = output_dir / "README.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    materialized_rows: list[dict[str, str]] = []
    for row in selected_rows:
        image_filename = f"{row['id_code']}.png"
        source_image_path = train_images_dir / image_filename
        if not source_image_path.exists():
            raise FileNotFoundError(f"Missing source image for sample dataset: {source_image_path}")

        destination_image_path = images_dir / image_filename
        shutil.copy2(source_image_path, destination_image_path)

        materialized_rows.append(
            {
                "id_code": row["id_code"],
                "diagnosis": row["diagnosis"],
                "image_filename": image_filename,
                "image_path": f"images/{image_filename}",
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(materialized_rows)

    readme_path.write_text(
        "\n".join(
            [
                "APTOS 2019 small deterministic sample dataset",
                "",
                f"Source CSV: {train_csv_path}",
                f"Source images: {train_images_dir}",
                "Selection rule: first 2 rows per diagnosis class after sorting by diagnosis and id_code.",
                f"Total samples: {len(materialized_rows)}",
                "CSV columns: id_code, diagnosis, image_filename, image_path",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return SampleDataset(
        output_dir=output_dir,
        csv_path=csv_path,
        images_dir=images_dir,
        rows=materialized_rows,
    )

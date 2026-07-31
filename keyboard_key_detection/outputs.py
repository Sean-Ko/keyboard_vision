import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from config import OUTPUT_DIR
from image_io import list_input_images


def draw_result(image, key_records):
    output = image.copy()
    for item in key_records:
        center = (int(round(float(item["cx"]))), int(round(float(item["cy"]))))
        cv2.circle(output, center, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return output


def safe_filename_part(text):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(text))


def build_training_record(key_records, model_name, input_image_path, raw_image_path, annotated_image_path, image_shape):
    height, width = image_shape[:2]
    channels = image_shape[2] if len(image_shape) == 3 else 1
    source_path = Path(input_image_path) if input_image_path is not None else None
    manual_sources = {"manual_added", "manual_fine_tuned"}

    annotations = []
    for index, item in enumerate(key_records, start=1):
        cx = round(float(item["cx"]), 2)
        cy = round(float(item["cy"]), 2)
        source = str(item.get("source", "unknown"))
        annotations.append(
            {
                "annotation_id": item.get("key_id", f"key_{index:03d}"),
                "label": item.get("key_name", f"key_{index:03d}"),
                "type": "key_center",
                "point": {"x": cx, "y": cy},
                "normalized_point": {
                    "x": round(cx / max(float(width), 1.0), 6),
                    "y": round(cy / max(float(height), 1.0), 6),
                },
                "source": source,
                "confidence": round(float(item.get("confidence", 0.0)), 4),
                "template": {
                    "x": float(item.get("template_x", 0.0)),
                    "y": float(item.get("template_y", 0.0)),
                    "width_unit": float(item.get("width_unit", 1.0)),
                    "height_unit": float(item.get("height_unit", 1.0)),
                },
            }
        )

    return {
        "schema_version": "keyboard_keypoints.v1",
        "task": "keyboard_key_center_detection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": str(model_name),
        "image": {
            "file_name": source_path.name if source_path else None,
            "source_path": str(source_path) if source_path else None,
            "raw_image_path": str(raw_image_path),
            "annotated_image_path": str(annotated_image_path),
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
        },
        "annotation_summary": {
            "type": "point_annotations",
            "coordinate_system": "pixel_xy_origin_top_left",
            "count": len(annotations),
            "manual_review_used": any(item["source"] in manual_sources for item in annotations),
            "dataset_split": "unassigned",
        },
        "annotations": annotations,
    }


def save_outputs(raw_image, annotated_image, key_records, model_name, input_image_path=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model_name = safe_filename_part(model_name)
    if input_image_path is None:
        prefix = f"{timestamp}_{safe_model_name}"
    else:
        safe_image_name = safe_filename_part(Path(input_image_path).stem)
        prefix = f"{timestamp}_{safe_image_name}_{safe_model_name}"

    raw_image_path = OUTPUT_DIR / f"{prefix}_raw.png"
    annotated_image_path = OUTPUT_DIR / f"{prefix}_calibrated.png"
    json_path = OUTPUT_DIR / f"{prefix}_keys.json"

    cv2.imwrite(str(raw_image_path), raw_image)
    cv2.imwrite(str(annotated_image_path), annotated_image)
    training_record = build_training_record(key_records, model_name, input_image_path, raw_image_path, annotated_image_path, raw_image.shape)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(training_record, file, ensure_ascii=False, indent=2)

    return raw_image_path, annotated_image_path, json_path


def clean_outputs(keep_latest_per_input=True):
    if not OUTPUT_DIR.exists():
        print(f"Output directory does not exist: {OUTPUT_DIR}")
        return 0

    keep = {OUTPUT_DIR / "README.md"}
    files = [path for path in OUTPUT_DIR.iterdir() if path.is_file()]

    if keep_latest_per_input:
        image_stems = [safe_filename_part(path.stem) for path in list_input_images()]
        for stem in image_stems:
            related = [path for path in files if f"_{stem}_" in path.name]
            for suffix in ("_raw.png", "_calibrated.png", "_keys.json"):
                matches = sorted((path for path in related if path.name.endswith(suffix)), key=lambda item: item.stat().st_mtime)
                if matches:
                    keep.add(matches[-1])

    removed = 0
    for path in files:
        if path in keep:
            continue
        path.unlink()
        removed += 1

    print(f"Cleaned output files: {removed}")
    return 0

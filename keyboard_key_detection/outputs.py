import json
from datetime import datetime
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


def save_outputs(image, key_records, model_name, input_image_path=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model_name = safe_filename_part(model_name)
    if input_image_path is None:
        prefix = f"{timestamp}_{safe_model_name}"
    else:
        safe_image_name = safe_filename_part(Path(input_image_path).stem)
        prefix = f"{timestamp}_{safe_image_name}_{safe_model_name}"

    image_path = OUTPUT_DIR / f"{prefix}_calibrated.png"
    json_path = OUTPUT_DIR / f"{prefix}_keys.json"

    cv2.imwrite(str(image_path), image)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(key_records, file, ensure_ascii=False, indent=2)

    return image_path, json_path


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
            for suffix in ("_calibrated.png", "_keys.json"):
                matches = sorted((path for path in related if path.name.endswith(suffix)), key=lambda item: item.stat().st_mtime)
                if matches:
                    keep.add(matches[-1])

        reports = sorted((path for path in files if path.name.endswith("_batch_report.json")), key=lambda item: item.stat().st_mtime)
        if reports:
            keep.add(reports[-1])

    removed = 0
    for path in files:
        if path in keep:
            continue
        path.unlink()
        removed += 1

    print(f"Cleaned output files: {removed}")
    return 0

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


CONFIG = {
    "canny_threshold1": 30,
    "canny_threshold2": 110,
    "min_area_ratio": 0.00075,
    "max_area_ratio": 0.04,
    "min_width_ratio": 0.018,
    "min_height_ratio": 0.045,
    "max_width_ratio": 0.45,
    "max_height_ratio": 0.22,
    "min_aspect_ratio": 0.45,
    "max_aspect_ratio": 10.0,
    "row_cluster_ratio": 0.55,
    "min_row_key_count": 6,
    "green_indicator_margin": 22,
}


def list_input_images():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_image_list(images):
    if not images:
        print(f"No images found in: {INPUT_DIR}")
        return
    print("Images in input/:")
    for index, path in enumerate(images, start=1):
        print(f"  {index}. {path.name}")


def resolve_image_path(image_arg):
    images = list_input_images()
    text = str(image_arg).strip()

    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(images):
            return images[index - 1]
        raise FileNotFoundError(f"Image index out of range: {index}")

    requested = Path(text)
    candidates = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend([INPUT_DIR / requested.name, PROJECT_DIR / requested, Path.cwd() / requested])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Input image not found. Tried:\n{tried}")


def choose_images(args):
    images = list_input_images()
    if args.list_images:
        print_image_list(images)
        return []

    if args.select:
        print_image_list(images)
        if not images:
            return []
        selected = input("Select image number, or press Enter for all: ").strip()
        if not selected:
            return images
        return [resolve_image_path(selected)]

    if args.image:
        return [resolve_image_path(args.image)]

    return images


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cv2.imread() failed to read image: {path}")
    return image


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(blurred)

    edges = cv2.Canny(enhanced, CONFIG["canny_threshold1"], CONFIG["canny_threshold2"])
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        4,
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    combined = cv2.bitwise_or(edges, adaptive)
    combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=1)

    return {"gray": gray, "enhanced": enhanced, "edges": edges, "adaptive": adaptive, "combined": combined}


def add_metrics(box):
    box["area"] = int(box["w"] * box["h"])
    box["cx"] = float(box["x"] + box["w"] / 2)
    box["cy"] = float(box["y"] + box["h"] / 2)
    return box


def find_keyboard_roi_from_gray(gray):
    mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    img_h, img_w = mask.shape[:2]

    row_counts = np.sum(mask > 0, axis=1)
    col_counts = np.sum(mask > 0, axis=0)
    rows = np.where(row_counts > img_w * 0.12)[0]
    cols = np.where(col_counts > img_h * 0.10)[0]
    if len(rows) and len(cols):
        y1, y2 = int(rows[0]), int(rows[-1])
        x1, x2 = int(cols[0]), int(cols[-1])
        return x1, y1, x2 - x1 + 1, y2 - y1 + 1

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, img_w, img_h
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def detect_contour_boxes(mask, image_shape, source, limits, rectangularity_min=None, retrieval=cv2.RETR_LIST):
    img_h, img_w = image_shape[:2]
    image_area = img_h * img_w
    contours, _ = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / float(h) if h else 0
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / float(area) if area else 0

        if area < image_area * limits["min_area"] or area > image_area * limits["max_area"]:
            continue
        if w < img_w * limits["min_w"] or h < img_h * limits["min_h"]:
            continue
        if w > img_w * limits["max_w"] or h > img_h * limits["max_h"]:
            continue
        if aspect < limits["min_aspect"] or aspect > limits["max_aspect"]:
            continue
        if rectangularity_min is not None and rectangularity < rectangularity_min:
            continue

        boxes.append(
            add_metrics(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "source": source,
                    "rectangularity": round(float(rectangularity), 3),
                }
            )
        )
    return boxes


def detect_edge_candidates(preprocessed, image_shape):
    return detect_contour_boxes(
        preprocessed["combined"],
        image_shape,
        "edge",
        {
            "min_area": CONFIG["min_area_ratio"],
            "max_area": CONFIG["max_area_ratio"],
            "min_w": CONFIG["min_width_ratio"],
            "min_h": CONFIG["min_height_ratio"],
            "max_w": CONFIG["max_width_ratio"],
            "max_h": CONFIG["max_height_ratio"],
            "min_aspect": CONFIG["min_aspect_ratio"],
            "max_aspect": CONFIG["max_aspect_ratio"],
        },
        rectangularity_min=0.08,
    )


def detect_brightness_candidates(preprocessed, image_shape):
    gray = preprocessed["gray"]
    mask = cv2.inRange(gray, 35, 215)
    roi_x, roi_y, roi_w, roi_h = find_keyboard_roi_from_gray(gray)
    roi_mask = np.zeros_like(mask)
    roi_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = 255
    mask = cv2.bitwise_and(mask, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    return detect_contour_boxes(
        mask,
        image_shape,
        "brightness",
        {
            "min_area": CONFIG["min_area_ratio"] * 0.65,
            "max_area": CONFIG["max_area_ratio"],
            "min_w": CONFIG["min_width_ratio"] * 0.85,
            "min_h": CONFIG["min_height_ratio"] * 0.85,
            "max_w": CONFIG["max_width_ratio"],
            "max_h": CONFIG["max_height_ratio"],
            "min_aspect": CONFIG["min_aspect_ratio"],
            "max_aspect": CONFIG["max_aspect_ratio"],
        },
        retrieval=cv2.RETR_EXTERNAL,
    )


def detect_light_keycap_candidates(preprocessed, image_shape):
    gray = preprocessed["gray"]
    if float(np.percentile(gray, 75)) < 145.0:
        return []

    boxes = []
    for threshold in (170, 185, 200):
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        boxes.extend(
            detect_contour_boxes(
                mask,
                image_shape,
                "light_keycap",
                {
                    "min_area": CONFIG["min_area_ratio"] * 0.75,
                    "max_area": CONFIG["max_area_ratio"],
                    "min_w": CONFIG["min_width_ratio"] * 0.85,
                    "min_h": CONFIG["min_height_ratio"] * 0.85,
                    "max_w": CONFIG["max_width_ratio"],
                    "max_h": CONFIG["max_height_ratio"],
                    "min_aspect": CONFIG["min_aspect_ratio"],
                    "max_aspect": CONFIG["max_aspect_ratio"],
                },
                rectangularity_min=0.18,
                retrieval=cv2.RETR_EXTERNAL,
            )
        )
    return boxes


def detect_bright_key_face_candidates(preprocessed, image_shape):
    gray = preprocessed["gray"]
    if float(np.percentile(gray, 75)) < 145.0:
        return []

    boxes = []
    for threshold in (135, 150, 165):
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        boxes.extend(
            detect_contour_boxes(
                mask,
                image_shape,
                "bright_key_face",
                {
                    "min_area": CONFIG["min_area_ratio"] * 0.35,
                    "max_area": CONFIG["max_area_ratio"] * 0.55,
                    "min_w": CONFIG["min_width_ratio"] * 0.45,
                    "min_h": CONFIG["min_height_ratio"] * 0.45,
                    "max_w": CONFIG["max_width_ratio"] * 0.28,
                    "max_h": CONFIG["max_height_ratio"] * 0.75,
                    "min_aspect": 0.45,
                    "max_aspect": 3.5,
                },
                rectangularity_min=0.18,
                retrieval=cv2.RETR_LIST,
            )
        )
    return boxes


def detect_midtone_keycap_candidates(preprocessed, image_shape):
    gray = preprocessed["gray"]
    if float(np.mean(gray)) > 150.0:
        return []

    boxes = []
    for low, high in ((25, 120), (45, 170), (65, 210)):
        mask = cv2.inRange(gray, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        boxes.extend(
            detect_contour_boxes(
                mask,
                image_shape,
                "midtone_keycap",
                {
                    "min_area": CONFIG["min_area_ratio"] * 0.55,
                    "max_area": CONFIG["max_area_ratio"],
                    "min_w": CONFIG["min_width_ratio"] * 0.75,
                    "min_h": CONFIG["min_height_ratio"] * 0.75,
                    "max_w": CONFIG["max_width_ratio"],
                    "max_h": CONFIG["max_height_ratio"],
                    "min_aspect": CONFIG["min_aspect_ratio"],
                    "max_aspect": CONFIG["max_aspect_ratio"],
                },
                rectangularity_min=0.18,
                retrieval=cv2.RETR_LIST,
            )
        )
    return boxes


def detect_tophat_candidates(preprocessed, image_shape):
    gray = preprocessed["gray"]
    keyboard_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    keyboard_mask = cv2.morphologyEx(keyboard_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)

    boxes = []
    for kernel_size in (31, 41, 51):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        top_hat = cv2.subtract(gray, cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel))
        top_hat = cv2.bitwise_and(top_hat, keyboard_mask)
        _, mask = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8), iterations=1)
        boxes.extend(
            detect_contour_boxes(
                mask,
                image_shape,
                "tophat",
                {
                    "min_area": 0.00035,
                    "max_area": 0.035,
                    "min_w": 0.012,
                    "min_h": 0.032,
                    "max_w": 0.32,
                    "max_h": 0.16,
                    "min_aspect": 0.45,
                    "max_aspect": 9.0,
                },
                rectangularity_min=0.28,
                retrieval=cv2.RETR_EXTERNAL,
            )
        )
    return boxes


def detect_legend_candidates(preprocessed, image_shape):
    boxes = detect_contour_boxes(
        preprocessed["combined"],
        image_shape,
        "legend",
        {
            "min_area": 0.00025,
            "max_area": 0.012,
            "min_w": 0.010,
            "min_h": 0.025,
            "max_w": 0.12,
            "max_h": 0.11,
            "min_aspect": 0.35,
            "max_aspect": 2.6,
        },
        rectangularity_min=0.48,
        retrieval=cv2.RETR_LIST,
    )

    expanded = []
    for box in boxes:
        w = max(box["w"], int(round(box["h"] * 0.92)))
        h = max(box["h"], int(round(w * 0.82)))
        expanded.append(
            add_metrics(
                {
                    "x": int(round(box["cx"] - w / 2)),
                    "y": int(round(box["cy"] - h / 2)),
                    "w": int(w),
                    "h": int(h),
                    "source": "legend",
                    "rectangularity": box["rectangularity"],
                }
            )
        )
    return expanded


def split_merged_edge_groups(boxes, image_shape):
    img_h, img_w = image_shape[:2]
    small = [
        box
        for box in boxes
        if box["w"] / max(float(box["h"]), 1.0) <= 1.8
        and img_w * 0.010 <= box["w"] <= img_w * 0.055
        and img_h * 0.025 <= box["h"] <= img_h * 0.11
    ]
    if not small:
        return []

    small_h = float(np.median([box["h"] for box in small]))
    split_boxes = []
    for group in boxes:
        if group.get("source") != "edge":
            continue
        aspect = group["w"] / max(float(group["h"]), 1.0)
        if aspect < 2.4 or aspect > 8.5:
            continue
        if group["h"] < small_h * 0.8 or group["h"] > small_h * 2.2:
            continue

        same_row = [box for box in small if abs(box["cy"] - group["cy"]) <= max(8.0, small_h * 0.75)]
        if len(same_row) < 2:
            continue

        row_h = float(np.median([box["h"] for box in same_row]))
        row_w = float(np.median([box["w"] for box in same_row]))
        ordered = sorted(same_row, key=lambda item: item["cx"])
        gaps = [
            right["cx"] - left["cx"]
            for left, right in zip(ordered, ordered[1:])
            if row_w * 1.15 <= right["cx"] - left["cx"] <= row_w * 2.4
        ]
        pitch = float(np.median(gaps)) if gaps else row_h * 1.35
        count = int(round(group["w"] / max(pitch, 1.0)))
        if count < 3 or count > 8:
            continue

        key_w = int(round(max(row_w, row_h * 0.82)))
        key_h = int(round(max(row_h, key_w * 0.92)))
        for index in range(count):
            cx = group["x"] + (index + 0.5) * group["w"] / count
            split_boxes.append(
                add_metrics(
                    {
                        "x": int(round(cx - key_w / 2)),
                        "y": int(round(group["cy"] - key_h / 2)),
                        "w": key_w,
                        "h": key_h,
                        "source": "edge_split",
                    }
                )
            )
    return split_boxes


def detect_key_candidates(preprocessed, image_shape):
    boxes = []
    boxes.extend(detect_edge_candidates(preprocessed, image_shape))
    boxes.extend(detect_brightness_candidates(preprocessed, image_shape))
    boxes.extend(detect_light_keycap_candidates(preprocessed, image_shape))
    boxes.extend(detect_bright_key_face_candidates(preprocessed, image_shape))
    boxes.extend(detect_tophat_candidates(preprocessed, image_shape))
    boxes.extend(detect_legend_candidates(preprocessed, image_shape))
    boxes.extend(split_merged_edge_groups(boxes, image_shape))
    return boxes


def intersection_area(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    return 0 if x2 <= x1 or y2 <= y1 else (x2 - x1) * (y2 - y1)


def filter_nested_boxes(boxes):
    if not boxes:
        return boxes

    typical_h = float(np.median([box["h"] for box in boxes]))
    keep = [True] * len(boxes)
    for i, small in enumerate(boxes):
        for j, large in enumerate(boxes):
            if i == j or not keep[i] or small["area"] >= large["area"]:
                continue
            overlap = intersection_area(small, large) / float(small["area"])
            preserve_split_candidate = (
                large.get("source") == "edge"
                and large["w"] >= small["w"] * 3.0
                and small["h"] >= typical_h * 0.65
                and small["h"] <= typical_h * 1.45
            )
            preserve_keycap_candidate = (
                small.get("source") in {"midtone_keycap", "bright_key_face"}
                and large["area"] >= small["area"] * 2.5
                and small["h"] >= typical_h * 0.55
                and small["h"] <= typical_h * 1.65
            )
            if overlap > 0.9 and not preserve_split_candidate and not preserve_keycap_candidate:
                keep[i] = False
    return [box for box, should_keep in zip(boxes, keep) if should_keep]


def estimate_key_size(boxes):
    if not boxes:
        return 1.0, 1.0
    heights = np.array([box["h"] for box in boxes], dtype=np.float32)
    key_h = float(np.median(heights))
    square_widths = [box["w"] for box in boxes if 0.75 <= box["w"] / max(key_h, 1) <= 1.5]
    unit_w = float(np.median(square_widths)) if square_widths else float(np.percentile([box["w"] for box in boxes], 35))
    return max(key_h, 1.0), max(unit_w, 1.0)


def score_key_candidate(box, typical_h, typical_w):
    source_score = {
        "brightness": 2.4,
        "light_keycap": 2.4,
        "light_keycap_split": 2.45,
        "bright_key_face": 2.5,
        "midtone_keycap": 2.35,
        "top_group_split": 2.45,
        "tophat": 2.0,
        "edge_split": 1.9,
        "edge": 1.5,
        "legend": 0.9,
    }.get(box.get("source"), 1.0)
    height_score = max(0.0, 1.0 - abs(box["h"] - typical_h) / max(typical_h, 1.0))
    width_ratio = box["w"] / max(typical_w, 1.0)
    if 0.65 <= width_ratio <= 1.55:
        width_score = 1.2
    elif width_ratio <= 2.4:
        width_score = 0.7
    elif width_ratio <= 6.5:
        width_score = -0.5
    else:
        width_score = -1.2
    aspect = box["w"] / max(float(box["h"]), 1.0)
    aspect_score = 0.8 if 0.55 <= aspect <= 1.8 else 0.2 if aspect <= 6.5 else -0.8
    return source_score + height_score + width_score + aspect_score + float(box.get("rectangularity", 0.45))


def suppress_duplicate_boxes(boxes):
    if not boxes:
        return []

    typical_h, typical_w = estimate_key_size(boxes)
    selected = []
    for box in sorted(boxes, key=lambda item: (score_key_candidate(item, typical_h, typical_w), -item["area"]), reverse=True):
        duplicate = False
        for existing in selected:
            inter = intersection_area(box, existing)
            if inter / float(max(1, min(box["area"], existing["area"]))) > 0.45:
                preserves_large_key = (
                    existing["area"] < box["area"] * 0.35
                    and box["h"] >= typical_h * 0.65
                    and box["h"] <= typical_h * 2.70
                    and box["w"] <= typical_w * 3.80
                )
                if preserves_large_key:
                    continue
                duplicate = True
                break
        if not duplicate:
            selected.append(box)
    return sorted(selected, key=lambda item: (item["cy"], item["cx"]))


def filter_key_sized_boxes(boxes):
    if len(boxes) < 5:
        return boxes

    heights = np.array([box["h"] for box in boxes], dtype=np.float32)
    widths = np.array([box["w"] for box in boxes], dtype=np.float32)
    median_h = float(np.median(heights))
    median_w = float(np.median(widths))
    lower = median_h * 0.55
    upper = median_h * 1.9

    filtered = []
    for box in boxes:
        vertical_key = box["h"] <= median_h * 2.7 and box["w"] <= median_w * 1.7 and box["h"] / max(box["w"], 1) >= 1.35
        large_enter_like = box["h"] <= median_h * 2.7 and median_w * 1.6 <= box["w"] <= median_w * 3.8
        if lower <= box["h"] <= upper or vertical_key or large_enter_like:
            filtered.append(box)
    return filtered


def split_merged_light_keycaps(boxes, image_shape):
    if len(boxes) < 10:
        return boxes

    img_h = image_shape[0]
    key_h, unit_w = estimate_key_size(boxes)
    split_boxes = []
    changed = False

    for box in boxes:
        width_ratio = box["w"] / max(unit_w, 1.0)
        height_ratio = box["h"] / max(key_h, 1.0)
        upper_rows = box["cy"] < img_h * 0.75
        looks_like_two_keys = (
            box.get("source") == "light_keycap"
            and upper_rows
            and 1.75 <= width_ratio <= 2.45
            and 0.75 <= height_ratio <= 1.30
        )

        if not looks_like_two_keys:
            split_boxes.append(box)
            continue

        changed = True
        key_w = int(round(unit_w))
        for index in range(2):
            cx = box["x"] + (index + 0.5) * box["w"] / 2
            split_boxes.append(
                add_metrics(
                    {
                        "x": int(round(cx - key_w / 2)),
                        "y": box["y"],
                        "w": key_w,
                        "h": box["h"],
                        "source": "light_keycap_split",
                    }
                )
            )

    return split_boxes if changed else boxes


def split_top_row_key_groups(boxes, image_shape):
    if len(boxes) < 10:
        return boxes

    img_h = image_shape[0]
    key_h, unit_w = estimate_key_size(boxes)
    output = []
    changed = False

    for box in boxes:
        width_ratio = box["w"] / max(unit_w, 1.0)
        height_ratio = box["h"] / max(key_h, 1.0)
        top_band = box["cy"] < img_h * 0.36
        source = box.get("source")
        should_split = (
            source in {"edge", "brightness", "midtone_keycap"}
            and top_band
            and 2.6 <= width_ratio <= 6.2
            and 0.75 <= height_ratio <= 1.85
        )

        if not should_split:
            output.append(box)
            continue

        count = int(round(box["w"] / max(unit_w * 1.2, 1.0)))
        if count < 3 or count > 5:
            output.append(box)
            continue

        changed = True
        key_w = int(round(min(unit_w, box["w"] / max(count, 1) * 0.78)))
        key_h_out = int(round(min(max(key_h, box["h"] * 0.62), box["h"] * 0.90)))
        for index in range(count):
            cx = box["x"] + (index + 0.5) * box["w"] / count
            output.append(
                add_metrics(
                    {
                        "x": int(round(cx - key_w / 2)),
                        "y": int(round(box["cy"] - key_h_out / 2)),
                        "w": key_w,
                        "h": key_h_out,
                        "source": "top_group_split",
                    }
                )
            )

    return output if changed else boxes


def prefer_keycaps_over_group_splits(boxes):
    if not boxes:
        return boxes

    key_h, unit_w = estimate_key_size(boxes)
    keycaps = [box for box in boxes if box.get("source") in {"midtone_keycap", "light_keycap", "bright_key_face", "edge_split"}]
    output = []

    for box in boxes:
        if box.get("source") == "top_group_split" and any(
            abs(box["cx"] - keycap["cx"]) <= unit_w * 0.55 and abs(box["cy"] - keycap["cy"]) <= key_h * 0.55
            for keycap in keycaps
        ):
            continue
        output.append(box)

    return output


def remove_top_row_gap_candidates(boxes, image_shape):
    if len(boxes) < 10:
        return boxes

    img_h = image_shape[0]
    key_h, unit_w = estimate_key_size(boxes)
    keycaps = [
        box
        for box in boxes
        if box.get("source") in {"midtone_keycap", "light_keycap", "bright_key_face", "edge_split"}
        and box["cy"] < img_h * 0.36
    ]
    output = []

    for box in boxes:
        source = box.get("source")
        if source not in {"top_group_split", "tophat"} or box["cy"] >= img_h * 0.36:
            output.append(box)
            continue

        left = [
            keycap
            for keycap in keycaps
            if 0 < box["cx"] - keycap["cx"] <= unit_w * 1.05
            and abs(box["cy"] - keycap["cy"]) <= key_h * 0.55
        ]
        right = [
            keycap
            for keycap in keycaps
            if 0 < keycap["cx"] - box["cx"] <= unit_w * 1.05
            and abs(box["cy"] - keycap["cy"]) <= key_h * 0.55
        ]
        tall_tophat = source == "tophat" and box["h"] > key_h * 1.30
        if left and right and (source == "top_group_split" or tall_tophat):
            continue

        output.append(box)

    return output


def filter_color_and_symbol_noise(image, boxes):
    filtered = []
    margin = CONFIG["green_indicator_margin"]
    for box in boxes:
        crop = image[box["y"] : box["y"] + box["h"], box["x"] : box["x"] + box["w"]]
        if crop.size == 0:
            continue

        mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
        if mean_g > mean_r + margin and mean_g > mean_b + margin:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        crop_mean = float(np.mean(gray))
        low_detail_indicator = box.get("source") == "edge" and crop_mean < 150.0 and float(np.std(gray)) < 12.0 and float(np.mean(edges > 0)) < 0.055
        bright_symbol = box.get("source") in {"legend", "tophat"} and crop_mean < 150.0 and float(np.mean(gray > 175)) > 0.22 and crop_mean > 115
        if not low_detail_indicator and not bright_symbol:
            filtered.append(box)
    return filtered


def cluster_rows(boxes, key_h):
    rows = []
    threshold = max(4.0, key_h * CONFIG["row_cluster_ratio"])

    for box in sorted(boxes, key=lambda item: item["cy"]):
        for row in rows:
            if abs(box["cy"] - row["mean"]) < threshold:
                row["boxes"].append(box)
                row["mean"] = float(np.mean([item["cy"] for item in row["boxes"]]))
                break
        else:
            rows.append({"mean": box["cy"], "boxes": [box]})

    keys = []
    for row_index, row in enumerate(sorted(rows, key=lambda item: item["mean"])):
        for col_index, box in enumerate(sorted(row["boxes"], key=lambda item: item["cx"])):
            item = dict(box)
            item["row"] = row_index
            item["col"] = col_index
            keys.append(item)
    return keys


def classify_key_type(box, unit_w):
    ratio = box["w"] / max(float(unit_w), 1.0)
    if 0.7 <= ratio < 1.35:
        key_type = "normal_key"
    elif 1.35 <= ratio < 2.2:
        key_type = "wide_key"
    elif 2.2 <= ratio < 4.0:
        key_type = "extra_wide_key"
    elif ratio >= 4.0:
        key_type = "spacebar"
    else:
        key_type = "unknown_key"
    box["unit_ratio"] = round(float(ratio), 3)
    box["key_type"] = key_type
    box["confidence"] = 0.75 if key_type != "unknown_key" else 0.4
    return box


def reindex_rows_and_columns(keys):
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)

    output = []
    for new_row, old_row in enumerate(sorted(rows)):
        for new_col, key in enumerate(sorted(rows[old_row], key=lambda item: item["cx"])):
            item = dict(key)
            item["row"] = new_row
            item["col"] = new_col
            output.append(item)
    return output


def filter_rows_and_outliers(keys):
    row_counts = {}
    for key in keys:
        row_counts[key["row"]] = row_counts.get(key["row"], 0) + 1
    valid_rows = {row for row, count in row_counts.items() if count >= CONFIG["min_row_key_count"]}
    keys = [key for key in keys if key["row"] in valid_rows and key["key_type"] != "unknown_key"]
    if not keys:
        return []

    typical_h = float(np.median([key["h"] for key in keys]))
    typical_w = float(np.median([key["w"] for key in keys]))

    def nearby_count(target):
        return sum(
            1
            for other in keys
            if other is not target
            and abs(target["cx"] - other["cx"]) <= typical_w * 2.4
            and abs(target["cy"] - other["cy"]) <= typical_h * 2.2
        )

    rows = {}
    for key in reindex_rows_and_columns(keys):
        rows.setdefault(key["row"], []).append(key)

    filtered = []
    for row_keys in rows.values():
        median_cy = float(np.median([key["cy"] for key in row_keys]))
        median_h = float(np.median([key["h"] for key in row_keys]))
        tolerance = max(5.0, median_h * 0.28)
        for key in row_keys:
            in_row = abs(key["cy"] - median_cy) <= tolerance
            vertical_key = key["h"] / max(key["w"], 1) >= 1.75 and key["h"] <= typical_h * 2.8
            large_enter_like = key["h"] <= typical_h * 3.0 and key["h"] >= typical_h * 1.65 and key["w"] >= typical_w * 1.45 and key["w"] <= typical_w * 3.8
            clustered_key = key["key_type"] == "normal_key" and nearby_count(key) >= 2
            if in_row or vertical_key or large_enter_like or clustered_key:
                filtered.append(key)
    return reindex_rows_and_columns(filtered)


def has_nearby_key(keys, cx, cy, x_tol, y_tol):
    return any(abs(key["cx"] - cx) <= x_tol and abs(key["cy"] - cy) <= y_tol for key in keys)


def infer_missing_keys(keys):
    normal_widths = [key["w"] for key in keys if key["key_type"] == "normal_key" and 0.75 <= key["unit_ratio"] <= 1.25]
    if len(keys) < 10 or not normal_widths:
        return keys

    unit_w = float(np.median(normal_widths))
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)

    gaps = []
    for row_keys in rows.values():
        ordered = sorted(row_keys, key=lambda item: item["cx"])
        gaps.extend(
            right["cx"] - left["cx"]
            for left, right in zip(ordered, ordered[1:])
            if unit_w * 1.15 <= right["cx"] - left["cx"] <= unit_w * 1.9
        )
    center_step = float(np.median(gaps)) if gaps else unit_w * 1.48
    completed = list(keys)

    completed = infer_row_gap_keys(completed, rows, center_step, unit_w)
    completed = infer_navigation_top_row(completed, center_step, unit_w)
    completed = infer_spacebar_modifier(completed, center_step, unit_w)
    completed = infer_vertical_edge_key(completed, center_step, unit_w)
    completed = infer_right_edge_column_keys(completed, center_step, unit_w)
    return reindex_rows_and_columns(completed)


def infer_row_gap_keys(keys, rows, center_step, unit_w):
    completed = [dict(key) for key in keys]
    for row, row_keys in rows.items():
        if row == min(rows) or len(row_keys) < 3:
            continue
        ordered = sorted(row_keys, key=lambda item: item["cx"])
        row_cy = float(np.median([item["cy"] for item in ordered]))
        row_h = float(np.median([item["h"] for item in ordered]))
        for left, right in zip(ordered, ordered[1:]):
            if abs(left["cy"] - row_cy) > max(5.0, row_h * 0.30) or abs(right["cy"] - row_cy) > max(5.0, row_h * 0.30):
                continue
            gap = right["cx"] - left["cx"]
            if not (center_step * 1.75 <= gap <= center_step * 2.65):
                continue
            if left["key_type"] != "normal_key" or right["key_type"] in ("wide_key", "extra_wide_key", "spacebar"):
                continue
            cx = left["cx"] + center_step
            if abs(right["cx"] - cx) >= center_step * 0.55:
                completed.append(make_inferred_key(cx, row_cy, unit_w, row_h, row, "row_gap_inferred", 0.52))
    return completed


def infer_navigation_top_row(keys, center_step, unit_w):
    completed = [dict(key) for key in keys]
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)
    if len(rows) < 3:
        return completed

    typical_h = float(np.median([key["h"] for key in keys]))
    keyboard_mid_x = float(np.median([key["cx"] for key in keys]))
    sorted_rows = sorted(rows)

    for upper_row, lower_row in zip(sorted_rows[1:], sorted_rows[2:]):
        upper_keys = sorted(rows[upper_row], key=lambda item: item["cx"])
        lower_keys = sorted(rows[lower_row], key=lambda item: item["cx"])
        upper_cy = float(np.median([key["cy"] for key in upper_keys]))

        for index in range(1, len(lower_keys) - 2):
            run = lower_keys[index : index + 3]
            run_gaps = [run[1]["cx"] - run[0]["cx"], run[2]["cx"] - run[1]["cx"]]
            if run[0]["cx"] < keyboard_mid_x or lower_keys[index]["cx"] - lower_keys[index - 1]["cx"] < center_step * 1.2:
                continue
            if not all(center_step * 0.75 <= gap <= center_step * 1.35 for gap in run_gaps):
                continue
            if any(has_nearby_key(upper_keys, key["cx"], upper_cy, center_step * 0.45, typical_h * 0.8) for key in run):
                continue

            left_upper = [key for key in upper_keys if key["cx"] < run[0]["cx"] - center_step * 0.65]
            right_upper = [key for key in upper_keys if key["cx"] > run[-1]["cx"] + center_step * 0.65]
            if not left_upper or not right_upper:
                continue
            upper_gap = min(key["cx"] for key in right_upper) - max(key["cx"] for key in left_upper)
            if not (center_step * 4.2 <= upper_gap <= center_step * 8.0):
                continue
            run_left = min(key["x"] for key in run)
            run_right = max(key["x"] + key["w"] for key in run)
            overlaps_existing_wide_key = any(
                key["key_type"] in ("wide_key", "extra_wide_key", "spacebar")
                and key["x"] < run_right
                and key["x"] + key["w"] > run_left
                for key in upper_keys
            )
            if overlaps_existing_wide_key:
                continue

            for key in run:
                completed.append(make_inferred_key(key["cx"], upper_cy, unit_w, typical_h, upper_row, "navigation_top_inferred", 0.54))
            return completed
    return completed


def infer_spacebar_modifier(keys, center_step, unit_w):
    completed = [dict(key) for key in keys]
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)

    typical_h = float(np.median([key["h"] for key in keys]))
    for row, row_keys in rows.items():
        spacebars = [key for key in row_keys if key["key_type"] == "spacebar"]
        if not spacebars:
            continue
        spacebar = max(spacebars, key=lambda item: item["w"])
        right_side = sorted([key for key in row_keys if key["cx"] > spacebar["cx"]], key=lambda item: item["cx"])
        if len(right_side) < 2:
            continue
        local_gaps = [b["cx"] - a["cx"] for a, b in zip(right_side, right_side[1:]) if center_step * 0.8 <= b["cx"] - a["cx"] <= center_step * 1.6]
        local_step = float(np.median(local_gaps)) if local_gaps else center_step * 1.25
        cx = right_side[0]["cx"] - local_step
        if cx > spacebar["x"] + spacebar["w"] + unit_w * 0.35 and not has_nearby_key(completed, cx, right_side[0]["cy"], local_step * 0.45, typical_h * 0.7):
            completed.append(make_inferred_key(cx, right_side[0]["cy"], unit_w, typical_h, row, "spacebar_modifier_inferred", 0.55))
        break
    return completed


def infer_vertical_edge_key(keys, center_step, unit_w):
    completed = [dict(key) for key in keys]
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)
    if len(rows) < 4:
        return completed

    row_stats = [
        {
            "row": row,
            "cy": float(np.median([key["cy"] for key in row_keys])),
            "max_x": max(key["cx"] for key in row_keys),
        }
        for row, row_keys in sorted(rows.items())
    ]
    global_right = float(np.median(sorted(stat["max_x"] for stat in row_stats)[-3:]))
    typical_h = float(np.median([key["h"] for key in keys]))

    for upper, lower in zip(row_stats, row_stats[1:]):
        if abs(upper["max_x"] - lower["max_x"]) > center_step * 0.45:
            continue
        if not (center_step * 0.70 <= global_right - upper["max_x"] <= center_step * 1.45):
            continue
        cy = (upper["cy"] + lower["cy"]) / 2
        h = abs(lower["cy"] - upper["cy"]) + typical_h
        if not has_nearby_key(completed, global_right, cy, center_step * 0.45, h * 0.45):
            completed.append(make_inferred_key(global_right, cy, unit_w, h, upper["row"], "vertical_edge_inferred", 0.54))
        break
    return completed


def infer_right_edge_column_keys(keys, center_step, unit_w):
    completed = [dict(key) for key in keys]
    typical_h = float(np.median([key["h"] for key in keys]))
    right_x = max(key["cx"] for key in keys)
    if right_x < 1500:
        return completed
    adjacent_column = [
        key
        for key in keys
        if right_x - center_step * 1.65 <= key["cx"] <= right_x - center_step * 0.45
    ]
    if len(adjacent_column) < 4:
        return completed

    y_values = sorted({round(key["cy"], 1) for key in adjacent_column})
    row = min(key["row"] for key in adjacent_column)
    for index in range(1, len(y_values) - 1, 2):
        upper_cy = float(y_values[index])
        lower_cy = float(y_values[index + 1])
        if lower_cy - upper_cy > typical_h * 2.2:
            continue
        cy = (upper_cy + lower_cy) / 2
        h = abs(lower_cy - upper_cy) + typical_h
        if not has_nearby_key(completed, right_x, cy, center_step * 0.50, h * 0.45):
            completed.append(make_inferred_key(right_x, cy, unit_w, h, row + index, "vertical_edge_inferred", 0.52))

    return completed


def make_inferred_key(cx, cy, w, h, row, source, confidence):
    return add_metrics(
        {
            "x": int(round(cx - w / 2)),
            "y": int(round(cy - h / 2)),
            "w": int(round(w)),
            "h": int(round(h)),
            "row": row,
            "col": 0,
            "unit_ratio": 1.0,
            "key_type": "normal_key",
            "confidence": confidence,
            "source": source,
        }
    )


def filter_blank_inferred_keys(image, keys):
    inferred_sources = {"row_gap_inferred", "spacebar_modifier_inferred", "navigation_top_inferred", "vertical_edge_inferred"}
    filtered = []
    for key in keys:
        if key.get("source") not in inferred_sources:
            filtered.append(key)
            continue
        crop = image[key["y"] : key["y"] + key["h"], key["x"] : key["x"] + key["w"]]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 30, 100) > 0))
        has_visible_key = float(np.mean(gray)) > 18.0 and (float(np.std(gray)) > 35.0 or edge_density > 0.03)
        if has_visible_key:
            filtered.append(key)
    return reindex_rows_and_columns(filtered)


def merge_right_edge_vertical_duplicates(keys):
    if len(keys) < 10:
        return keys

    typical_h = float(np.median([key["h"] for key in keys]))
    typical_w = float(np.median([key["w"] for key in keys]))
    right_edge = max(key["cx"] for key in keys)
    used = set()
    merged = []

    ordered = sorted(enumerate(keys), key=lambda item: (item[1]["cx"], item[1]["cy"]))
    for i, key in ordered:
        if i in used:
            continue

        partner = None
        if key["cx"] > right_edge - typical_w * 1.2 and key["key_type"] == "normal_key":
            for j, other in ordered:
                if j == i or j in used or other["key_type"] != "normal_key":
                    continue
                same_column = abs(other["cx"] - key["cx"]) <= typical_w * 0.45
                vertical_gap = abs(other["cy"] - key["cy"])
                if same_column and typical_h * 0.75 <= vertical_gap <= typical_h * 1.45:
                    partner = (j, other)
                    break

        if partner is None:
            merged.append(key)
            used.add(i)
            continue

        j, other = partner
        x1 = min(key["x"], other["x"])
        y1 = min(key["y"], other["y"])
        x2 = max(key["x"] + key["w"], other["x"] + other["w"])
        y2 = max(key["y"] + key["h"], other["y"] + other["h"])
        item = add_metrics(
            {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "row": min(key["row"], other["row"]),
                "col": 0,
                "unit_ratio": round(float((x2 - x1) / max(typical_w, 1.0)), 3),
                "key_type": "vertical_key",
                "confidence": min(float(key["confidence"]), float(other["confidence"])),
                "source": "vertical_duplicate_merged",
            }
        )
        merged.append(item)
        used.update({i, j})

    return reindex_rows_and_columns(merged)


def filter_sparse_rows(keys):
    row_counts = {}
    for key in keys:
        row_counts[key["row"]] = row_counts.get(key["row"], 0) + 1
    valid_rows = {row for row, count in row_counts.items() if count >= CONFIG["min_row_key_count"]}
    return reindex_rows_and_columns([key for key in keys if key["row"] in valid_rows])


def remove_top_status_row(keys):
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)
    if len(rows) < 3:
        return keys

    first_row = min(rows)
    second_row = sorted(rows)[1]
    first_keys = rows[first_row]
    second_keys = rows[second_row]
    first_count = len(first_keys)
    second_count = len(second_keys)
    first_span = max(key["cx"] for key in first_keys) - min(key["cx"] for key in first_keys)
    all_span = max(key["cx"] for key in keys) - min(key["cx"] for key in keys)
    first_cy = float(np.median([key["cy"] for key in first_keys]))
    second_cy = float(np.median([key["cy"] for key in second_keys]))
    typical_h = float(np.median([key["h"] for key in keys]))

    looks_like_status_row = (
        first_count <= max(8, second_count * 0.55)
        and first_span < all_span * 0.62
        and second_cy - first_cy > typical_h * 0.95
    )
    if not looks_like_status_row:
        return keys

    return reindex_rows_and_columns([key for key in keys if key["row"] != first_row])


def suppress_close_center_duplicates(keys):
    if len(keys) < 2:
        return keys

    typical_w = float(np.median([key["w"] for key in keys if key["key_type"] == "normal_key"] or [key["w"] for key in keys]))
    typical_h = float(np.median([key["h"] for key in keys if key["key_type"] == "normal_key"] or [key["h"] for key in keys]))

    def keep_score(key):
        source_score = {
            "bright_key_face": 4,
            "midtone_keycap": 4,
            "light_keycap": 4,
            "edge": 3,
            "brightness": 3,
            "tophat": 2,
            "legend": 1,
        }.get(key.get("source"), 2)
        size_score = 1.0 - min(1.0, abs(key["w"] - typical_w) / max(typical_w, 1.0))
        return source_score + size_score

    selected = []
    for key in sorted(keys, key=keep_score, reverse=True):
        if any(abs(key["cx"] - other["cx"]) <= typical_w * 0.32 and abs(key["cy"] - other["cy"]) <= typical_h * 0.65 for other in selected):
            continue
        selected.append(key)

    return reindex_rows_and_columns(selected)


def remove_large_gap_keys(image, keys):
    img_h, img_w = image.shape[:2]
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)

    filtered = []
    for row_keys in rows.values():
        median_h = float(np.median([key["h"] for key in row_keys]))
        median_w = float(np.median([key["w"] for key in row_keys]))
        for key in row_keys:
            source = key.get("source")
            top_right_edge_status_area = (
                source == "edge"
                and key["cx"] > img_w * 0.78
                and key["cy"] < img_h * 0.34
                and key["w"] >= median_w * 1.35
                and key["h"] >= median_h * 1.20
            )
            top_right_label_status_area = (
                source == "legend"
                and key["row"] == min(rows)
                and key["cx"] > img_w * 0.75
                and key["cy"] < img_h * 0.42
                and key["h"] <= median_h * 1.25
                and key["w"] <= median_w * 1.45
            )
            if top_right_edge_status_area or top_right_label_status_area:
                continue

            large_overlay = source in {"legend", "tophat"} and (
                key["unit_ratio"] > 2.35 or key["h"] > median_h * 1.45 or key["w"] > median_w * 2.35
            )
            if large_overlay:
                left = [other for other in row_keys if other is not key and other["cx"] < key["cx"] and key["cx"] - other["cx"] <= median_w * 1.35]
                right = [other for other in row_keys if other is not key and other["cx"] > key["cx"] and other["cx"] - key["cx"] <= median_w * 1.35]
                tall_top_tophat = source == "tophat" and key["row"] == min(rows) and key["h"] > median_h * 1.35
                if (left and right) or tall_top_tophat:
                    continue

            crop = image[key["y"] : key["y"] + key["h"], key["x"] : key["x"] + key["w"]]
            if crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                edge_density = float(np.mean(cv2.Canny(gray, 30, 100) > 0))
                low_detail_body = source == "edge" and float(np.mean(gray)) < 150.0 and float(np.std(gray)) < 18.0 and edge_density < 0.025
                if low_detail_body:
                    continue

            filtered.append(key)

    return reindex_rows_and_columns(filtered)


def expand_bottom_row_spacebars(keys):
    if not keys:
        return keys

    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)
    bottom_row = max(rows)
    row_keys = sorted(rows[bottom_row], key=lambda item: item["cx"])
    if len(row_keys) < 5:
        return keys

    normal_widths = [key["w"] for key in keys if key["key_type"] == "normal_key"]
    unit_w = float(np.median(normal_widths)) if normal_widths else float(np.median([key["w"] for key in keys]))
    adjusted = [dict(key) for key in keys]

    for left, key, right in zip(row_keys, row_keys[1:], row_keys[2:]):
        right_gap = right["x"] - (key["x"] + key["w"])
        if key["w"] < unit_w * 2.8 and right_gap > unit_w * 2.4 and key["w"] > unit_w * 1.4:
            for item in adjusted:
                if item["row"] == key["row"] and item["col"] == key["col"]:
                    new_right = right["x"] - max(2, int(round(unit_w * 0.10)))
                    item["w"] = int(max(item["w"], new_right - item["x"]))
                    add_metrics(item)
                    item["unit_ratio"] = round(float(item["w"] / max(unit_w, 1.0)), 3)
                    if item["unit_ratio"] >= 4.0:
                        item["key_type"] = "spacebar"
                    break

    return reindex_rows_and_columns(adjusted)


def align_row_centers(keys):
    rows = {}
    for key in keys:
        rows.setdefault(key["row"], []).append(key)

    aligned = []
    for row_keys in rows.values():
        median_cy = float(np.median([key["cy"] for key in row_keys]))
        median_h = float(np.median([key["h"] for key in row_keys]))
        for key in row_keys:
            item = dict(key)
            if item["h"] <= median_h * 1.8 and item["key_type"] != "vertical_key":
                item["cy"] = median_cy
                item["y"] = int(round(median_cy - item["h"] / 2))
            aligned.append(item)

    return reindex_rows_and_columns(aligned)


def finalize_candidates(image, candidates, preprocessed):
    boxes = filter_nested_boxes(candidates)
    boxes = suppress_duplicate_boxes(boxes)
    boxes = filter_color_and_symbol_noise(image, boxes)
    boxes = filter_key_sized_boxes(boxes)
    boxes = split_top_row_key_groups(boxes, image.shape)
    boxes = prefer_keycaps_over_group_splits(boxes)
    boxes = remove_top_row_gap_candidates(boxes, image.shape)
    boxes = split_merged_light_keycaps(boxes, image.shape)

    key_h, unit_w = estimate_key_size(boxes)
    keys = [classify_key_type(key, unit_w) for key in cluster_rows(boxes, key_h)]
    keys = filter_rows_and_outliers(keys)
    keys = infer_missing_keys(keys)
    keys = filter_blank_inferred_keys(image, keys)
    keys = merge_right_edge_vertical_duplicates(keys)
    keys = filter_sparse_rows(keys)
    keys = remove_top_status_row(keys)
    keys = remove_large_gap_keys(image, keys)
    keys = expand_bottom_row_spacebars(keys)
    keys = align_row_centers(keys)
    keys = suppress_close_center_duplicates(keys)
    return keys, len(candidates), preprocessed


def detect_keys(image):
    preprocessed = preprocess_image(image)
    candidates = detect_key_candidates(preprocessed, image.shape)
    result = finalize_candidates(image, candidates, preprocessed)

    gray_mean = float(np.mean(preprocessed["gray"]))
    if len(result[0]) < 80 and gray_mean < 150.0:
        expanded_candidates = candidates + detect_midtone_keycap_candidates(preprocessed, image.shape)
        expanded_result = finalize_candidates(image, expanded_candidates, preprocessed)
        if len(expanded_result[0]) > len(result[0]):
            return expanded_result

    return result


def draw_detection_result(image, keys, draw_labels=False):
    output = image.copy()
    radius = max(5, int(round(max(image.shape[:2]) / 320)))
    for key in keys:
        point = (int(round(key["cx"])), int(round(key["cy"])))
        cv2.circle(output, point, radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        if draw_labels:
            cv2.putText(output, f'{key["row"]}-{key["col"]}', (point[0] + 7, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    return output


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text)


def key_rows(keys):
    rows = []
    for index, key in enumerate(sorted(keys, key=lambda item: (item["row"], item["col"])), start=1):
        rows.append(
            {
                "key_id": f"key_{index:03d}",
                "row": int(key["row"]),
                "col": int(key["col"]),
                "x": int(key["x"]),
                "y": int(key["y"]),
                "w": int(key["w"]),
                "h": int(key["h"]),
                "cx": round(float(key["cx"]), 2),
                "cy": round(float(key["cy"]), 2),
                "unit_ratio": float(key["unit_ratio"]),
                "key_type": key["key_type"],
                "confidence": float(key["confidence"]),
            }
        )
    return rows


def create_run_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suffix = 2
    while run_dir.exists():
        run_dir = OUTPUT_DIR / f"{run_dir.name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_outputs(image, keys, image_path, output_dir, model_name=None):
    label = safe_name(model_name or image_path.stem)
    prefix = label

    image_output = output_dir / f"{prefix}_detected.png"
    json_output = output_dir / f"{prefix}_keys.json"

    cv2.imwrite(str(image_output), image)
    rows = key_rows(keys)

    with json_output.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    return image_output, json_output


def save_debug_images(preprocessed, image_path, output_dir):
    debug_dir = output_dir / "debug" / image_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, debug_image in preprocessed.items():
        cv2.imwrite(str(debug_dir / f"{name}.png"), debug_image)


def print_summary(image_path, image, raw_count, keys, paths):
    img_h, img_w = image.shape[:2]
    row_counts = {}
    for key in keys:
        row_counts[key["row"]] = row_counts.get(key["row"], 0) + 1

    print(f"\nDetection summary: {image_path.name}")
    print(f"- Image size: {img_w} x {img_h}")
    print(f"- Raw candidates: {raw_count}")
    print(f"- Filtered keys: {len(keys)}")
    print(f"- Rows: {', '.join(str(row_counts[row]) for row in sorted(row_counts))}")
    print(f"- Output image: {paths[0]}")
    print(f"- JSON: {paths[1]}")


def process_image(image_path, args, output_dir):
    image = load_image(image_path)
    keys, raw_count, preprocessed = detect_keys(image)
    output = draw_detection_result(image, keys, args.draw_labels)
    model_name = args.model if args.model and len(args.images) == 1 else image_path.stem
    paths = save_outputs(output, keys, image_path, output_dir, model_name)
    if args.debug:
        save_debug_images(preprocessed, image_path, output_dir)
    print_summary(image_path, image, raw_count, keys, paths)
    return len(keys)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect keyboard key centers with an OpenCV geometric baseline.")
    parser.add_argument("images", nargs="*", help="Optional image names, paths, or input/ indexes.")
    parser.add_argument("--image", help="Single image name, path, or input/ index.")
    parser.add_argument("--model", help="Output label. Defaults to the image file name.")
    parser.add_argument("--select", action="store_true", help="Pick an image from input/ interactively.")
    parser.add_argument("--list-images", action="store_true", help="List images in input/ and exit.")
    parser.add_argument("--draw-labels", action="store_true", help="Draw row-column labels on the output image.")
    parser.add_argument("--debug", action="store_true", help="Save preprocessing masks under output/debug/.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.images and args.image:
        print("Error: use positional images or --image, not both.")
        return 1

    try:
        image_paths = [resolve_image_path(item) for item in args.images] if args.images else choose_images(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if args.list_images:
        return 0
    if not image_paths:
        print(f"No images to process. Put images in: {INPUT_DIR}")
        return 1

    args.images = image_paths
    args.output_dir = create_run_dir()
    print(f"Output run folder: {args.output_dir}")
    failed = 0
    for image_path in image_paths:
        try:
            process_image(image_path, args, args.output_dir)
        except (FileNotFoundError, ValueError) as exc:
            failed += 1
            print(f"Error processing {image_path}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

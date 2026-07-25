import detector_core


def run_auto_detection(image):
    keys, raw_count, _preprocessed = detector_core.detect_keys(image)
    sorted_keys = sorted(keys, key=lambda item: (item["row"], item["col"]))

    records = []
    for index, key in enumerate(sorted_keys, start=1):
        row = int(key["row"])
        col = int(key["col"])
        records.append(
            {
                "key_id": f"key_{index:03d}",
                "key_name": f"row{row}_col{col}",
                "template_x": float(col),
                "template_y": float(row),
                "width_unit": float(key.get("unit_ratio", 1.0)),
                "height_unit": 1.0,
                "cx": round(float(key["cx"]), 2),
                "cy": round(float(key["cy"]), 2),
                "source": f"auto_opencv:{key.get('source', 'unknown')}",
                "confidence": float(key.get("confidence", 0.8)),
            }
        )

    return records, {"raw_count": raw_count, "keys": len(records)}

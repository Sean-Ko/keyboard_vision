import cv2
import numpy as np

from config import HEADER_HEIGHT


def region_bucket(record):
    x = float(record.get("norm_x", 0.5))
    y = float(record.get("norm_y", 0.5))
    horizontal = "left" if x < 0.33 else "center" if x < 0.66 else "right"
    vertical = "top" if y < 0.33 else "middle" if y < 0.66 else "bottom"
    return f"{vertical}_{horizontal}"


def make_display_canvas(image, header_lines):
    canvas = np.zeros((image.shape[0] + HEADER_HEIGHT, image.shape[1], 3), dtype=np.uint8)
    canvas[HEADER_HEIGHT:, :] = image
    cv2.rectangle(canvas, (0, 0), (image.shape[1], HEADER_HEIGHT), (24, 24, 24), -1)
    for index, line in enumerate(header_lines[:3]):
        cv2.putText(
            canvas,
            line,
            (14, 30 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def find_nearest_record_index(records, x, y, max_distance=22):
    if not records:
        return None
    distances = [float(np.hypot(float(item["cx"]) - x, float(item["cy"]) - y)) for item in records]
    index = int(np.argmin(distances))
    return index if distances[index] <= max_distance else None


def fine_tune_key_records(image, initial_key_records):
    window_name = "Keyboard Key Center Review"
    records = [dict(item) for item in initial_key_records]
    original_records = [dict(item) for item in initial_key_records]
    deleted_stack = []
    manual_add_counter = 1
    state = {"selected": None, "dragging": False, "add_mode": False}

    def add_manual_record(x, y):
        nonlocal manual_add_counter
        key_name = f"manual_add_{manual_add_counter:03d}"
        manual_add_counter += 1
        records.append(
            {
                "key_id": key_name,
                "key_name": key_name,
                "template_x": round(float(x), 2),
                "template_y": round(float(y), 2),
                "width_unit": 1.0,
                "height_unit": 1.0,
                "cx": round(float(x), 2),
                "cy": round(float(y), 2),
                "source": "manual_added",
                "confidence": 1.0,
            }
        )
        state["selected"] = len(records) - 1

    def mouse_callback(event, x, y, _flags, _param):
        image_x = float(x)
        image_y = float(y - HEADER_HEIGHT)
        if image_y < 0:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if state["add_mode"]:
                add_manual_record(image_x, image_y)
                state["add_mode"] = False
                return
            state["selected"] = find_nearest_record_index(records, image_x, image_y)
            state["dragging"] = state["selected"] is not None
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] and state["selected"] is not None:
            selected = records[state["selected"]]
            selected["cx"] = round(image_x, 2)
            selected["cy"] = round(image_y, 2)
            selected["source"] = "manual_fine_tuned"
            selected["confidence"] = 1.0
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        selected_name = "none"
        if state["selected"] is not None and 0 <= state["selected"] < len(records):
            selected_name = records[state["selected"]]["key_name"]
        mode_text = "ADD MODE: click image to add one missing center." if state["add_mode"] else "Drag only the wrong dot."
        display = make_display_canvas(
            image,
            [
                f"Auto detection result. {mode_text} Selected: {selected_name}",
                "A add | D delete selected | U undo delete | R reset | Enter save | Esc cancel",
            ],
        )

        for index, item in enumerate(records):
            center = (int(round(float(item["cx"]))), int(round(float(item["cy"]) + HEADER_HEIGHT)))
            color = (0, 255, 255) if index == state["selected"] else (0, 255, 0)
            radius = 6 if index == state["selected"] else 4
            cv2.circle(display, center, radius, color, -1, lineType=cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Cancelled by user.")
        if key in (13, 10):
            cv2.destroyWindow(window_name)
            return records
        if key in (ord("a"), ord("A")):
            state["add_mode"] = not state["add_mode"]
            state["dragging"] = False
        if key in (ord("d"), ord("D")) and state["selected"] is not None:
            deleted_stack.append((state["selected"], dict(records[state["selected"]])))
            del records[state["selected"]]
            state["selected"] = None
        if key in (ord("u"), ord("U")) and deleted_stack:
            restore_index, restored = deleted_stack.pop()
            restore_index = min(restore_index, len(records))
            records.insert(restore_index, restored)
            state["selected"] = restore_index
        if key in (ord("r"), ord("R")):
            records = [dict(item) for item in original_records]
            deleted_stack.clear()
            manual_add_counter = 1
            state["selected"] = None
            state["dragging"] = False
            state["add_mode"] = False

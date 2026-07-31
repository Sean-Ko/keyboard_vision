import argparse

from config import OUTPUT_DIR
from detector import run_auto_detection
from gui_review import fine_tune_key_records
from image_io import list_input_images, load_image, select_input_image
from outputs import clean_outputs, draw_result, save_outputs


def run_single_image(image_arg, model_name, no_gui=False):
    image_path, image = load_image(image_arg)
    key_records, detection_info = run_auto_detection(image)
    if not key_records:
        print("Automatic detection found 0 keys. Improve detector_core.py before using manual review.")
        return 1, None

    print(
        "Automatic detection complete: "
        f"{detection_info['keys']} keys, {detection_info['raw_count']} raw candidates."
    )

    if not no_gui:
        try:
            key_records = fine_tune_key_records(image, key_records)
        except KeyboardInterrupt as exc:
            print(exc)
            return 1, None

    annotated_image = draw_result(image, key_records)
    raw_output, annotated_output, json_output = save_outputs(image, annotated_image, key_records, model_name, image_path)

    result = {
        "input_image": str(image_path),
        "model": model_name,
        "keys": len(key_records),
        "raw_candidates": detection_info["raw_count"],
        "raw_image": str(raw_output),
        "annotated_image": str(annotated_output),
        "json": str(json_output),
    }

    print("Done.")
    print(f"- Input image: {image_path}")
    print(f"- Model: {model_name}")
    print(f"- Keys: {len(key_records)}")
    print(f"- Raw image: {raw_output}")
    print(f"- Annotated image: {annotated_output}")
    print(f"- JSON: {json_output}")
    return 0, result


def run_batch_input(model_name):
    images = list_input_images()
    if not images:
        print("No input images found.")
        return 1

    exit_code = 0
    for image_path in images:
        print(f"\n=== Batch image: {image_path.name} ===")
        code, result = run_single_image(image_path, model_name, no_gui=True)
        exit_code = max(exit_code, code)

    print(f"\nBatch complete. Outputs are in: {OUTPUT_DIR}")
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV keyboard key-center detection with optional manual review.")
    parser.add_argument("--image", default=None, help="Input keyboard image path. If omitted, an input/ image picker is shown.")
    parser.add_argument("--select-image", action="store_true", help="Show an input/ image picker before running detection.")
    parser.add_argument("--model", default="keyboard", help='Output model name used in filenames. Default: "keyboard".')
    parser.add_argument("--no-gui", action="store_true", help="Run automatic detection and save outputs without opening GUI.")
    parser.add_argument("--batch-input", action="store_true", help="Run every supported image in input/ without GUI.")
    parser.add_argument("--clean-outputs", action="store_true", help="Remove old output files while keeping README and latest result per input image.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clean_outputs:
        return clean_outputs()
    if args.batch_input:
        return run_batch_input(args.model)

    image_arg = select_input_image() if args.select_image or args.image is None else args.image
    code, _result = run_single_image(image_arg, args.model, no_gui=args.no_gui)
    return code

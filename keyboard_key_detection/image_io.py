from pathlib import Path

import cv2

from config import IMAGE_EXTENSIONS, INPUT_DIR, PROJECT_DIR


def load_image(image_path):
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Cannot read image: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return path, image


def list_input_images():
    if not INPUT_DIR.exists():
        return []
    return sorted(path for path in INPUT_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def select_input_image():
    images = list_input_images()
    if not images:
        raise FileNotFoundError(f"No input images found in: {INPUT_DIR}")

    print("\nImages in input/:")
    for index, path in enumerate(images, start=1):
        print(f"  {index}. {path.name}")

    while True:
        choice = input("Select image number, or press Enter for 1: ").strip()
        if not choice:
            return images[0]
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(images):
                return images[index - 1]
        print(f"Please enter a number from 1 to {len(images)}.")

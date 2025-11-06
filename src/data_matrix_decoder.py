import os
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
from pylibdmtx.pylibdmtx import decode as dmtx_decode
from .utils import draw_results, create_error_record, create_empty_record

def preprocess_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Apply multiple preprocessing techniques to improve detection.
    Returns list of processed versions.
    """
    versions = []

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. Original grayscale
    versions.append(gray)

    # 2. Thresholding (adaptive)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    versions.append(thresh)

    # 3. Inverted
    versions.append(cv2.bitwise_not(gray))

    # 4. Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    versions.append(closed)

    # 5. Sharpened
    blurred = cv2.GaussianBlur(gray, (0,0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    versions.append(sharpened)

    # 6. High contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    versions.append(enhanced)

    return versions

def decode_datamatrix_from_image(
    image_path: str,
    timeout_ms: int = 1000,
    max_count: Optional[int] = None
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Decode all Data Matrix codes in an image with preprocessing.
    Returns list of (data, bounding_box)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    preprocessed_images = preprocess_image(image)
    all_results = []

    for proc_img in preprocessed_images:
        try:
            # Convert to PIL for pylibdmtx
            pil_img = Image.fromarray(proc_img)

            # Decode with pylibdmtx
            decoded = dmtx_decode(
                pil_img,
                timeout=timeout_ms,
                max_count=max_count,
                shrinkage=1  # Try full size first
            )

            # Extract bounding boxes from OpenCV image
            for d in decoded:
                data = d.data.decode('utf-8')
                rect = d.rect  # pylibdmtx Rect: left, top, width, height

                # Convert to (x, y, w, h)
                box = (rect.left, rect.top, rect.width, rect.height)
                all_results.append((data, box))

        except Exception as e:
            continue  # Try next preprocessing

    # Remove duplicates
    seen = set()
    unique_results = []
    for data, box in all_results:
        key = (data, box)
        if key not in seen:
            seen.add(key)
            unique_results.append((data, box))

    return unique_results



def process_image(
    image_path: str,
    output_dir: str,
    draw_boxes: bool,
    timeout_ms: int
) -> List[dict]:
    """Process single image and return results"""
    try:
        results = decode_datamatrix_from_image(image_path, timeout_ms=timeout_ms)
        records = []

        if results:
            print(f"Found {len(results)} Data Matrix code(s) in {image_path}")

            for i, (data, box) in enumerate(results):
                x, y, w, h = box
                record = {
                    'filename': Path(image_path).name,
                    'code_index': i + 1,
                    'data': data,
                    'x': x, 'y': y, 'width': w, 'height': h
                }
                records.append(record)
                print(f"  [{i+1}] {data} @ ({x}, {y}, {w}x{h})")

            if draw_boxes:
                out_img_path = os.path.join(output_dir, f"annotated_{Path(image_path).name}")
                image = cv2.imread(image_path)
                draw_results(image, results, out_img_path)
                print(f"  Annotated image saved: {out_img_path}")

        else:
            print(f"No Data Matrix found in {image_path}")
            records.append(create_empty_record(Path(image_path).name))

        return records

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return [create_error_record(Path(image_path).name, e)]

def process_batch(
    image_files: List[Path],
    output_dir: Path,
    draw_boxes: bool = False,
    timeout_ms: int = 2000
) -> List[dict]:
    """Process multiple images and return all results"""
    all_records = []
    for img_path in image_files:
        records = process_image(
            str(img_path),
            str(output_dir),
            draw_boxes=draw_boxes,
            timeout_ms=timeout_ms
        )
        all_records.extend(records)
    return all_records
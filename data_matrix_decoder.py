import os
import cv2
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
from pylibdmtx.pylibdmtx import decode as dmtx_decode

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

def draw_results(image: np.ndarray, results: List[Tuple[str, Tuple[int, int, int, int]]], output_path: str):
    """Draw bounding boxes and text on image"""
    img_with_boxes = image.copy()
    for data, (x, y, w, h) in results:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, data[:20], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, img_with_boxes)

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
            records.append({
                'filename': Path(image_path).name,
                'code_index': 0,
                'data': '',
                'x': '', 'y': '', 'width': '', 'height': ''
            })

        return records

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return [{
            'filename': Path(image_path).name,
            'code_index': 0,
            'data': f'ERROR: {e}',
            'x': '', 'y': '', 'width': '', 'height': ''
        }]

def main():
    parser = argparse.ArgumentParser(description="Powerful Data Matrix Decoder")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    parser.add_argument("--csv", default="datamatrix_results.csv", help="CSV output file")
    parser.add_argument("--draw", action="store_true", help="Draw boxes on output images")
    parser.add_argument("--timeout", type=int, default=2000, help="Decoding timeout in ms (default: 2000)")
    parser.add_argument("--max-count", type=int, help="Max number of codes to decode per image")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Collect image files
    if input_path.is_dir():
        image_files = list(input_path.glob("*.png")) + \
                      list(input_path.glob("*.jpg")) + \
                      list(input_path.glob("*.jpeg")) + \
                      list(input_path.glob("*.bmp")) + \
                      list(input_path.glob("*.tiff"))
    else:
        image_files = [input_path]

    if not image_files:
        print("No image files found.")
        return

    # Process all images
    all_records = []
    for img_path in image_files:
        records = process_image(
            str(img_path),
            str(output_dir),
            draw_boxes=args.draw,
            timeout_ms=args.timeout
        )
        all_records.extend(records)

    # Save CSV
    csv_path = Path(args.csv)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'code_index', 'data', 'x', 'y', 'width', 'height']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nDecoding complete. Results saved to {csv_path}")
    if args.draw:
        print(f"Annotated images saved in {output_dir}")

if __name__ == "__main__":
    main()
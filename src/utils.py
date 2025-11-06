import os
import cv2
import csv
from pathlib import Path
from typing import List, Tuple
import numpy as np


def draw_results(image: np.ndarray, results: List[Tuple[str, Tuple[int, int, int, int]]], output_path: str):
    """Draw bounding boxes and text on image"""
    img_with_boxes = image.copy()
    for data, (x, y, w, h) in results:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, data[:20], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, img_with_boxes)


def save_results_to_csv(records: List[dict], csv_path: Path):
    """Save results to CSV file"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'code_index', 'data', 'x', 'y', 'width', 'height']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def collect_image_files(input_path: Path) -> List[Path]:
    """Collect all image files from input path"""
    if input_path.is_dir():
        image_files = list(input_path.glob("*.png")) + \
                      list(input_path.glob("*.jpg")) + \
                      list(input_path.glob("*.jpeg")) + \
                      list(input_path.glob("*.bmp")) + \
                      list(input_path.glob("*.tiff"))
    else:
        image_files = [input_path]
    
    return image_files


def create_error_record(filename: str, error: Exception) -> dict:
    """Create error record for failed processing"""
    return {
        'filename': filename,
        'code_index': 0,
        'data': f'ERROR: {error}',
        'x': '', 'y': '', 'width': '', 'height': ''
    }


def create_empty_record(filename: str) -> dict:
    """Create empty record when no codes found"""
    return {
        'filename': filename,
        'code_index': 0,
        'data': '',
        'x': '', 'y': '', 'width': '', 'height': ''
    }
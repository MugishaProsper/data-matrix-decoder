#!/usr/bin/env python3
"""
Data Matrix Decoder Server
Main entry point for the data matrix decoding service
"""

import argparse
from pathlib import Path
from src.data_matrix_decoder import process_batch
from src.utils import collect_image_files, save_results_to_csv


def main():
    """Main entry point for the Data Matrix Decoder"""
    parser = argparse.ArgumentParser(description="Powerful Data Matrix Decoder")
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    parser.add_argument("--csv", default="datamatrix_results.csv", help="CSV output file")
    parser.add_argument("--draw", action="store_true", help="Draw boxes on output images")
    parser.add_argument("--timeout", type=int, default=2000, help="Decoding timeout in ms (default: 2000)")
    parser.add_argument("--max-count", type=int, help="Max number of codes to decode per image")

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Collect image files
    image_files = collect_image_files(input_path)
    
    if not image_files:
        print("No image files found.")
        return

    print(f"Processing {len(image_files)} image(s)...")

    # Process all images
    all_records = process_batch(
        image_files=image_files,
        output_dir=output_dir,
        draw_boxes=args.draw,
        timeout_ms=args.timeout
    )

    # Save results to CSV
    csv_path = Path(args.csv)
    save_results_to_csv(all_records, csv_path)

    print(f"\nDecoding complete. Results saved to {csv_path}")
    if args.draw:
        print(f"Annotated images saved in {output_dir}")


if __name__ == "__main__":
    main()
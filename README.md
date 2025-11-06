# Data Matrix Decoder

A powerful Python service for decoding Data Matrix codes from images with multiple preprocessing techniques and both CLI and web server interfaces.

## Features

- Multiple image preprocessing techniques for better detection
- Support for various image formats (PNG, JPG, JPEG, BMP, TIFF)
- CLI interface for batch processing
- Web server API for integration
- Bounding box detection and visualization
- CSV export of results

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Process a single image:
```bash
python main.py path/to/image.jpg
```

Process a directory of images:
```bash
python main.py path/to/images/ --draw --csv results.csv
```

Options:
- `--output` / `-o`: Output directory for annotated images (default: results)
- `--csv`: CSV output file (default: datamatrix_results.csv)
- `--draw`: Draw bounding boxes on output images
- `--timeout`: Decoding timeout in milliseconds (default: 2000)

### Web Server

Start the Flask server:
```bash
python server.py
```

The server runs on `http://localhost:5000` by default.

#### API Endpoints

**POST /decode**
- Upload an image file to decode Data Matrix codes
- Form parameters:
  - `file`: Image file (required)
  - `timeout`: Timeout in ms (optional, default: 2000)
  - `draw_boxes`: Create annotated image (optional, default: false)

**GET /**
- Health check endpoint

**GET /download/<filename>**
- Download annotated images

#### Example API Usage

```bash
curl -X POST -F "file=@image.jpg" -F "draw_boxes=true" http://localhost:5000/decode
```

## Environment Variables

- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: false)

## Project Structure

```
├── main.py              # CLI entry point
├── server.py            # Web server
├── requirements.txt     # Dependencies
├── src/
│   ├── __init__.py
│   ├── data_matrix_decoder.py  # Core decoding functions
│   └── utils.py         # Utility functions
├── uploads/             # Temporary upload directory
└── results/             # Output directory
```

## Dependencies

- OpenCV (opencv-python)
- pylibdmtx
- Pillow
- NumPy
- Flask (for web server)
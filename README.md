# YOLO to COCO Format Converter

A Python utility for converting YOLO polygon annotations to COCO JSON format for object detection and instance segmentation tasks.

## Overview

This script converts polygon annotations from YOLO format (normalized coordinates in .txt files) to the COCO JSON format, which is widely used in computer vision datasets. The conversion includes proper handling of:

- Polygon segmentation coordinates
- Bounding box calculations
- Area calculations using the shoelace formula
- Proper category mapping

## Features

- Converts YOLO polygon annotations to COCO format
- Automatically calculates bounding boxes from polygon points
- Computes polygon areas using the shoelace formula
- Preserves image-annotation relationships
- Handles multiple categories
- Validates input data (checks for valid polygon point counts)

## Requirements

- Python 3.x
- PIL (Pillow) for image processing

## Installation

```bash
# Clone the repository
git clone https://github.com/Shalessa/Yolo-Segmentation-to-COCO-format.git
cd Yolo-Segmentation-to-COCO-format

# Install dependencies
pip install pillow
```

## Usage

1. Update the following variables in the script:
   - `images_dir`: Path to your image directory
   - `labels_dir`: Path to your YOLO annotation directory
   - `output_dir`: Where the COCO JSON will be saved
   - `categories`: Define your object categories

2. If your images are not PNG format, update the file extension filter in the script.

3. Run the script:
   ```python
   python yolo_segmentation_to_coco.py
   ```

## Input Format

The script expects:

- Images in PNG format (or modify the extension in the code)
- YOLO format annotation files (.txt) with the same base filename as the corresponding image
- Each line in the annotation file should be: `class_id x1 y1 x2 y2 x3 y3 ...`
  - `class_id`: Integer representing the object category
  - `x1 y1 x2 y2 ...`: Normalized polygon coordinates (values between 0 and 1)

## Output Format

The script produces a single `coco_format.json` file with the following structure:

```json
{
  "info": {"description": "my-project-name"},
  "images": [
    {"id": 1, "width": 640, "height": 480, "file_name": "image1.png"},
    ...
  ],
  "annotations": [
    {
      "id": 0,
      "iscrowd": 0,
      "image_id": 1,
      "category_id": 0,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, width, height],
      "area": 1234.5
    },
    ...
  ],
  "categories": [
    {"id": 0, "name": "category_name"},
    ...
  ]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

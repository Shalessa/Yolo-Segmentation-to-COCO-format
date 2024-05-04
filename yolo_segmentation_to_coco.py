import json
import os
import re
from PIL import Image

# Helper function to calculate the area of a polygon using the shoelace formula
def calculate_polygon_area(polygon):
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return 0.5 * abs(sum(x[i] * y[i + 1] - y[i] * x[i + 1] for i in range(-1, len(x) - 1)))

# Define the directory containing the images and YOLO annotations
images_dir = 'path/to/your/images'
labels_dir = 'path/to/your/labels'

# Specify output directory for the COCO formatted JSON
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Get list of images and labels
# Change png to jpg/jpeg if needed
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

# Define categories
# Might add more if needed
categories = [{'id': 0, 'name': 'tape'}]

# Initialize COCO dataset structure
coco_data = {
    "info": {"description": "my-project-name"},
    "images": [],
    "annotations": [],
    "categories": categories
}

# Dictionary to map image filenames to unique IDs
image_id_map = {name: idx for idx, name in enumerate(image_files, start=1)}

# Annotation counters
ann_id = 0

for image_filename in image_files:
    image_path = os.path.join(images_dir, image_filename)
    image = Image.open(image_path)
    width, height = image.size
    image_id = image_id_map[image_filename]

    # Add image information to COCO dataset
    coco_data['images'].append({
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": image_filename
    })

    # Corresponding label file
    label_filename = image_filename.replace('.png', '.txt')
    label_path = os.path.join(labels_dir, label_filename)

    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file:
                segments = line.strip().split()
                class_id = int(segments[0])
                points = segments[1:]
                if len(points) % 2 != 0:
                    print(f"Warning: Ignored annotation with odd number of coordinates in {label_filename}")
                    continue
                polygon = [[float(points[i]), float(points[i+1])] for i in range(0, len(points), 2)]
                abs_polygon = [[x * width, y * height] for x, y in polygon]
                flat_polygon = [coord for xy in abs_polygon for coord in xy]
                bbox_x = min(xy[0] for xy in abs_polygon)
                bbox_y = min(xy[1] for xy in abs_polygon)
                bbox_w = max(xy[0] for xy in abs_polygon) - bbox_x
                bbox_h = max(xy[1] for xy in abs_polygon) - bbox_y
                area = calculate_polygon_area(abs_polygon)

                coco_data['annotations'].append({
                    "id": ann_id,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [flat_polygon],
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area": area
                })
                ann_id += 1

# Save the COCO data to a file
output_path = os.path.join(output_dir, 'coco_format.json')
with open(output_path, 'w') as f:
    json_str = json.dumps(coco_data, separators=(',', ':'))
    json_str = re.sub(r'\s+', '', json_str)  # Remove all whitespace
    f.write(json_str)

print("Conversion complete!")

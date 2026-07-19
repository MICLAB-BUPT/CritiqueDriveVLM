"""Stitch the 6 nuScenes surround-view cameras of each sample into a single
2x3 image, used as the visual input for every stage of CritiqueDriveVLM.

Edit the paths in the CONFIG section below before running.
"""
import os
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG (edit these) =================
input_json_path = "/path/to/data/DriveLMMo1_TRAIN.json"          # DriveLMM-o1 annotation json
base_image_dir = "/path/to/data/nuscenes"                        # nuScenes image root
savepath_img = Path("/path/to/data/nuscenes/stitched_output")    # output directory
savepath_img.mkdir(parents=True, exist_ok=True)

# Define the correct camera order
cameras = [
"CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
"CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"
]

# Load JSON
with open(input_json_path, 'r') as f:
    dataset = json.load(f)

# Helper to extract the prefix (removes last _digit)
def get_prefix_id(idx):
    return "_".join(idx.split("_")[:-1])

# Track already processed prefixes
already_processed = set(p.stem for p in savepath_img.glob("*.png"))

# Function to extract camera name from filename
def extract_camera_name(path):
    return path.split("/")[1] if len(path.split("/")) > 1 else None

# Main loop
for item in tqdm(dataset):
    idx = item["idx"]
    image_paths = item["image"]
    prefix_id = get_prefix_id(idx)
    output_path = savepath_img / f"{prefix_id}.png"

    if prefix_id in already_processed or output_path.exists():
        continue  # Skip if already processed

# Match each image path to the correct camera using exact name match
    cam_to_image = {cam: None for cam in cameras}
    for img_path in image_paths:
        cam_name = extract_camera_name(img_path)
        if cam_name in cam_to_image:
            cam_to_image[cam_name] = os.path.join(base_image_dir, img_path)

# Create a 2x3 stitched image plot
    fig, axes = plt.subplots(2, 3, figsize=(60, 24), gridspec_kw={'wspace': 0, 'hspace': 0})

    for i, cam in enumerate(cameras):
        img_path = cam_to_image[cam]
        ax = axes[0, i] if i < 3 else axes[1, i - 3]

        if img_path is None or not os.path.exists(img_path):
            print(f"Warning: Missing image for {cam} in {idx}")
            ax.axis('off')
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis('off')

    # Save and close
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    already_processed.add(prefix_id)
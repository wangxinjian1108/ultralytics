#!/bin/bash

# Activate the YOLO environment
conda activate yolo

pth_path=${1:-"./pths/yolo11n-seg.pt"}
img_path=${2:-"./tmp/person.png"}

# Run the segmentation script
python scripts/segment.py --model $pth_path --image $img_path
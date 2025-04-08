#!/bin/bash

# Activate the YOLO environment

pth_path=${1:-"./pths/yolo11n-seg.pt"}
img_path=${2:-"./tmp/person.png"}
output_dir=${3:-"./tmp/output"}
class_names=${4:-"person"}
height=${5:-"544"}
width=${6:-"960"}

# Run the segmentation script
python scripts/segment.py --model $pth_path --source $img_path --output-dir $output_dir --class-names $class_names --imgsz $height $width
#!/bin/bash

# Activate the YOLO environment

pth_path=${1:-"./pths/yoloe-11l-seg-pf.pt"}
img_path=${2:-"./tmp/front.png"}
output_dir=${3:-"./tmp/output"}
class_names=${4:-"person.car.suv.truck.traffic_cone"}
height=${5:-"544"}
width=${6:-"960"}

# Run the segmentation script
python scripts/segment.py --model $pth_path --source $img_path --output-dir $output_dir --class-names $class_names --imgsz $height $width
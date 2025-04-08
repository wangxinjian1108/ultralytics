#!/bin/bash
# Script to run YOLO segmentation on a directory of images and create a video
# Usage: ./segment_dir.sh <model_path> <input_dir> <output_dir> [fps] [class_names]

set -e

# source conda activate yolo

# Default values
MODEL=${1:-"yoloe-11l-seg-pf.pt"}
INPUT_DIR=${2:-"/home/xinjian/Code/VAutoLabelerCore/data/1027_20241230T102329_pdb-l4e-c0010_20_570to578/raw_images/front_left_camera"}
OUTPUT_DIR=${3:-"$(pwd)/tmp/video"}
FPS=${4:-10}
CLASS_NAMES=${5:-"car.suv.truck.traffic_cone.person.bicycle.motorcycle"}
IMAGE_SIZE=${6:-544 960}

# Check if required directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file '$MODEL' not found"
    exit 1
fi

# Run the Python script
echo "Running segmentation on images in $INPUT_DIR"
echo "Output will be saved to $OUTPUT_DIR"
echo "Using model: $MODEL"
echo "Video FPS: $FPS"

# Build the command with optional class names
CMD="python scripts/segment_dir.py --model \"$MODEL\" --input-dir \"$INPUT_DIR\" --output-dir \"$OUTPUT_DIR\" --fps $FPS --imgsz $IMAGE_SIZE"

# Add class names if provided
if [ ! -z "$CLASS_NAMES" ]; then
    echo "Using class names: $CLASS_NAMES"
    CMD="$CMD --class-names \"$CLASS_NAMES\""
fi

# Execute the command
eval $CMD

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Segmentation completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Video saved as: $OUTPUT_DIR/segmentation_video.mp4"
else
    echo "Error: Segmentation failed"
    exit 1
fi 
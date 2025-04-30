#!/bin/bash
# Ultralytics YOLO Training Script
# This script automates the process of training YOLO models with custom datasets

# Default configuration
MODEL="yolo12n.pt"
DATA="/home/xinjian.wang/ultralytics/scripts/custom_object365.yaml"
EPOCHS=100
IMG_SIZE=640
BATCH=192
DEVICE="5,6,7,9"
WORKERS=8
NAME="yolo-12n"
PROJECT="runs/train"
INFERENCE=""
CUSTOM_DIR="/mnt/juicefs/xinjian/open_dataset/object365_tiny"

# Display help message
show_help() {
    echo "Ultralytics YOLO Training Script"
    echo "--------------------------------"
    echo "Usage: ./train.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -m, --model MODEL          Model file path (.pt or .yaml) [default: yolo12n.pt]"
    echo "  -d, --data YAML            Dataset configuration file (.yaml) [default: object365.yaml]"
    echo "  -e, --epochs EPOCHS        Number of training epochs [default: 100]"
    echo "  -i, --img-size SIZE        Input image size [default: 640]"
    echo "  -b, --batch BATCH          Batch size [default: 16]"
    echo "  --device DEVICE            Device to use (e.g., 0,1,2,3, cpu) [default: auto]"
    echo "  -w, --workers WORKERS      Number of dataloader workers [default: 8]"
    echo "  -n, --name NAME            Experiment name [default: exp]"
    echo "  -p, --project PROJECT      Project directory [default: runs/train]"
    echo "  --inference PATH           Run inference on this image/video after training"
    echo "  --dir PATH                 Custom directory for downloading Object365 dataset"
    echo ""
    echo "Examples:"
    echo "  ./train.sh                                  # Train with default settings"
    echo "  ./train.sh --epochs 50 --batch 8            # Train for 50 epochs with batch size 8"
    echo "  ./train.sh --dir /data/datasets/object365   # Download dataset to custom directory"
    echo "  ./train.sh --device 0,1                     # Train using GPUs 0 and 1"
    echo "  ./train.sh --inference test.jpg             # Run inference after training"
    echo ""
    echo "Note: The Object365 dataset is very large (~712GB). Make sure you have sufficient disk space."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        show_help
        ;;
    -m | --model)
        MODEL="$2"
        shift 2
        ;;
    -d | --data)
        DATA="$2"
        shift 2
        ;;
    -e | --epochs)
        EPOCHS="$2"
        shift 2
        ;;
    -i | --img-size)
        IMG_SIZE="$2"
        shift 2
        ;;
    -b | --batch)
        BATCH="$2"
        shift 2
        ;;
    --device)
        DEVICE="$2"
        shift 2
        ;;
    -w | --workers)
        WORKERS="$2"
        shift 2
        ;;
    -n | --name)
        NAME="$2"
        shift 2
        ;;
    -p | --project)
        PROJECT="$2"
        shift 2
        ;;
    --inference)
        INFERENCE="$2"
        shift 2
        ;;
    --dir)
        CUSTOM_DIR="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        ;;
    esac
done

# Check if Python and pip are installed
if ! command -v python &>/dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if ultralytics is installed
if ! python -c "import ultralytics" &>/dev/null; then
    echo "Ultralytics package not found. Installing..."
    pip install ultralytics
fi

# Set Objects365 download directory if specified
if [ -n "$CUSTOM_DIR" ]; then
    export OBJECTS365_DIR="$CUSTOM_DIR"
    echo "Setting Objects365 dataset download directory to: $CUSTOM_DIR"
fi

# Build the command
CMD="python scripts/train.py --model $MODEL --data $DATA --epochs $EPOCHS --imgsz $IMG_SIZE --batch $BATCH --workers $WORKERS --name $NAME --project $PROJECT"

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$INFERENCE" ]; then
    CMD="$CMD --inference $INFERENCE"
fi

# Print the command being executed
echo "Executing: $CMD"
echo "Starting training..."

# Execute the training command
$CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $PROJECT/$NAME"
else
    echo "Training failed with an error."
    exit 1
fi

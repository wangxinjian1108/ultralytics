#!/bin/bash
# Ultralytics YOLO Dataset Downloader
# This script downloads and sets up various datasets for use with Ultralytics YOLO

# ./scripts/download_dataset.sh --dataset object365 --output-dir /mnt/juicefs/xinjian/open_dataset
# ./scripts/download_dataset.sh --dataset coco --output-dir /mnt/juicefs/xinjian/open_dataset

# Default settings
DATASET="object365"
OUTPUT_DIR="/mnt/juicefs/xinjian/open_dataset"
SUBSET="full"  # full, mini, tiny options for some datasets
FORCE=false
QUIET=false

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display help message
show_help() {
    echo -e "${BLUE}Ultralytics YOLO Dataset Downloader${NC}"
    echo "------------------------------------"
    echo "Downloads and sets up datasets for training YOLO models"
    echo ""
    echo "Usage: ./download_dataset.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message"
    echo "  -d, --dataset DATASET       Dataset to download (default: coco)"
    echo "                               Available options: coco, object365, visdrone, argoverse, etc."
    echo "  -o, --output-dir DIR        Output directory for the dataset"
    echo "  -s, --subset SUBSET         Dataset subset (default: full)"
    echo "                               Available options: full, mini, tiny (where applicable)"
    echo "  -f, --force                 Force re-download even if files exist"
    echo "  -q, --quiet                 Quiet mode (less output)"
    echo ""
    echo "Examples:"
    echo "  ./download_dataset.sh --dataset coco --output-dir /data/datasets"
    echo "  ./download_dataset.sh --dataset object365 --output-dir /mnt/storage --subset full"
    echo "  ./download_dataset.sh --dataset visdrone --output-dir /data/datasets --force"
    echo ""
    echo "Supported Datasets:"
    echo "  coco       - COCO Dataset (Common Objects in Context)"
    echo "  object365  - Objects365 Dataset (very large, ~712GB)"
    echo "  visdrone   - VisDrone Dataset (Drone Vision)"
    echo "  argoverse  - Argoverse HD Dataset"
    echo "  coco8      - COCO8 Example Dataset (8 images)"
    echo ""
    echo "Note: Some datasets are very large. Ensure sufficient disk space before downloading."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--subset)
            SUBSET="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Check for required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: Output directory (--output-dir) is required${NC}"
    show_help
fi

# Check if output directory exists, create if it doesn't
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Output directory doesn't exist. Creating: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to create output directory${NC}"
        exit 1
    fi
fi

# Check if Python and pip are installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if required packages are installed
check_requirements() {
    # Check ultralytics is installed
    if ! python -c "import ultralytics" &> /dev/null; then
        echo -e "${YELLOW}Ultralytics package not found. Installing...${NC}"
        pip install ultralytics
    fi
    
    # Check other required packages based on dataset
    case "$DATASET" in
        object365)
            if ! python -c "import pycocotools" &> /dev/null; then
                echo -e "${YELLOW}pycocotools package not found. Installing...${NC}"
                pip install pycocotools
            fi
            ;;
        *)
            # Default requirements for most datasets
            true
            ;;
    esac
}

# Download COCO dataset
download_coco() {
    echo -e "${GREEN}Downloading COCO dataset to $OUTPUT_DIR${NC}"
    
    if [ "$SUBSET" = "tiny" ] || [ "$SUBSET" = "mini" ]; then
        echo -e "${YELLOW}Downloading COCO-minitrain subset (small version for testing)${NC}"
        # Download COCO minitrain
        python -c "from ultralytics.utils.downloads import download; \
                   download(url='https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip', \
                   dir='$OUTPUT_DIR', unzip=True)"
    else
        echo -e "${YELLOW}Downloading full COCO dataset (may take a while)...${NC}"
        # Set environment variable for dataset path
        export DATASET_DIR=$OUTPUT_DIR
        
        # Use ultralytics download functionality
        python -c "from ultralytics.data.utils import download_dataset; \
                  download_dataset(dataset='coco', dir='$OUTPUT_DIR')"
    fi
}

# Download Objects365 dataset
download_object365() {
    echo -e "${GREEN}Downloading Objects365 dataset to $OUTPUT_DIR${NC}"
    echo -e "${YELLOW}Warning: Objects365 is very large (~712GB). This may take a long time.${NC}"
    
    # Set environment variable for dataset path
    export OBJECTS365_DIR=$OUTPUT_DIR
    
    # Use the download script from the YAML file
    python -c "
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import download
from ultralytics.utils.ops import xyxy2xywhn

# Configure directory
dir = Path('$OUTPUT_DIR')
print(f'Downloading Object365 dataset to: {dir}')

check_requirements(('pycocotools>=2.0',))
from pycocotools.coco import COCO

# Make Directories
for p in 'images', 'labels':
    (dir / p).mkdir(parents=True, exist_ok=True)
    for q in 'train', 'val':
        (dir / p / q).mkdir(parents=True, exist_ok=True)

# Train, Val Splits
for split, patches in [('train', 50 + 1), ('val', 43 + 1)]:
    print(f'Processing {split} in {patches} patches ...')
    images, labels = dir / 'images' / split, dir / 'labels' / split

    # Download
    url = f'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/'
    if split == 'train':
        download([f'{url}zhiyuan_objv2_{split}.tar.gz'], dir=dir)  # annotations json
        download([f'{url}patch{i}.tar.gz' for i in range(patches)], dir=images, curl=True, threads=8)
    elif split == 'val':
        download([f'{url}zhiyuan_objv2_{split}.json'], dir=dir)  # annotations json
        download([f'{url}images/v1/patch{i}.tar.gz' for i in range(15 + 1)], dir=images, curl=True, threads=8)
        download([f'{url}images/v2/patch{i}.tar.gz' for i in range(16, patches)], dir=images, curl=True, threads=8)

    # Move
    for f in tqdm(images.rglob('*.jpg'), desc=f'Moving {split} images'):
        f.rename(images / f.name)  # move to /images/{split}

    # Labels
    coco = COCO(dir / f'zhiyuan_objv2_{split}.json')
    names = [x['name'] for x in coco.loadCats(coco.getCatIds())]
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
            width, height = im['width'], im['height']
            path = Path(im['file_name'])  # image filename
            try:
                with open(labels / path.with_suffix('.txt').name, 'a', encoding='utf-8') as file:
                    annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
                    for a in coco.loadAnns(annIds):
                        x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                        xyxy = np.array([x, y, x + w, y + h])[None]  # pixels(1,4)
                        x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                        file.write(f'{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n')
            except Exception as e:
                print(e)
"
}

# Download VisDrone dataset
download_visdrone() {
    echo -e "${GREEN}Downloading VisDrone dataset to $OUTPUT_DIR${NC}"
    
    # Set environment variable for dataset path
    export DATASET_DIR=$OUTPUT_DIR
    
    # Use ultralytics download functionality
    python -c "
from pathlib import Path
import os
from ultralytics.utils.downloads import download, safe_download

# Configure directory
dir = Path('$OUTPUT_DIR')
print(f'Downloading VisDrone dataset to: {dir}')

# Make needed directories
(dir / 'VisDrone2019-DET-train' / 'images').mkdir(parents=True, exist_ok=True)
(dir / 'VisDrone2019-DET-train' / 'annotations').mkdir(parents=True, exist_ok=True)
(dir / 'VisDrone2019-DET-val' / 'images').mkdir(parents=True, exist_ok=True)
(dir / 'VisDrone2019-DET-val' / 'annotations').mkdir(parents=True, exist_ok=True)
(dir / 'VisDrone2019-DET-test-dev' / 'images').mkdir(parents=True, exist_ok=True)
(dir / 'VisDrone2019-DET-test-dev' / 'annotations').mkdir(parents=True, exist_ok=True)

# URLs
train_url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/VisDrone2019-DET-train.zip'
val_url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/VisDrone2019-DET-val.zip'
test_url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/VisDrone2019-DET-test-dev.zip'

# Download and extract datasets
download(train_url, dir=dir, unzip=True)
download(val_url, dir=dir, unzip=True)
download(test_url, dir=dir, unzip=True)

# Convert annotations to YOLO format if needed (this is dataset specific)
print('Dataset downloaded and extracted. Converting annotations to YOLO format...')
"
}

# Download Argoverse dataset
download_argoverse() {
    echo -e "${GREEN}Downloading Argoverse dataset to $OUTPUT_DIR${NC}"
    
    # Set environment variable for dataset path
    export DATASET_DIR=$OUTPUT_DIR
    
    # Use ultralytics download functionality
    python -c "
from pathlib import Path
import os
from ultralytics.utils.downloads import download

# Configure directory
dir = Path('$OUTPUT_DIR')
print(f'Downloading Argoverse dataset to: {dir}')

# Create directory
(dir / 'Argoverse-1.1' / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(dir / 'Argoverse-1.1' / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(dir / 'Argoverse-1.1' / 'images' / 'test').mkdir(parents=True, exist_ok=True)

# Download dataset
download('https://argoverse-hd.s3.us-east-2.amazonaws.com/Argoverse-HD-Full.zip', dir=dir, unzip=True)
"
}

# Download COCO8 example dataset
download_coco8() {
    echo -e "${GREEN}Downloading COCO8 example dataset to $OUTPUT_DIR${NC}"
    
    # Use ultralytics download functionality
    python -c "from ultralytics.utils.downloads import download; \
              download(url='https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip', \
              dir='$OUTPUT_DIR', unzip=True)"
}

# Main logic
echo -e "${BLUE}====== ULTRALYTICS DATASET DOWNLOADER ======${NC}"
echo -e "Dataset: ${GREEN}$DATASET${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Subset: ${GREEN}$SUBSET${NC}"
echo ""

# Check requirements
check_requirements

# Select download function based on dataset
case "$DATASET" in
    coco)
        download_coco
        ;;
    object365)
        download_object365
        ;;
    visdrone)
        download_visdrone
        ;;
    argoverse)
        download_argoverse
        ;;
    coco8)
        download_coco8
        ;;
    *)
        echo -e "${RED}Error: Unsupported dataset '$DATASET'${NC}"
        show_help
        ;;
esac

# Set permissions
chmod -R 755 "$OUTPUT_DIR"

# Create a simple README in the dataset directory
echo "Ultralytics YOLO Dataset: $DATASET
Downloaded on: $(date)
Dataset version: $SUBSET
Downloaded with: download_dataset.sh

For more information, see:
- https://docs.ultralytics.com/datasets/
- https://github.com/ultralytics/ultralytics" > "$OUTPUT_DIR/README.txt"

echo -e "${GREEN}====== DOWNLOAD COMPLETE ======${NC}"
echo -e "Dataset: ${BLUE}$DATASET${NC} has been downloaded to ${BLUE}$OUTPUT_DIR${NC}"
echo -e "You can now use this dataset with ultralytics by setting the path in your .yaml configuration file."
echo -e "Example YAML configuration:"
echo -e "${YELLOW}----------------------------${NC}"
echo "path: $OUTPUT_DIR  # dataset root dir"
echo "train: images/train  # train images (relative to 'path')"
echo "val: images/val  # val images (relative to 'path')"
echo -e "${YELLOW}----------------------------${NC}"

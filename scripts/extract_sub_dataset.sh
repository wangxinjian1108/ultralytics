#!/bin/bash

set -e

python scripts/create_tiny_dataset.py \
    --dataset-path /mnt/juicefs/xinjian/open_dataset/object365 \
    --output-dir /mnt/juicefs/xinjian/open_dataset/object365_tiny \
    --selected-classes 5,34,49,55,65,58,46,0,66 \
    --total-images 1000

#!/bin/bash
# refer: https://docs.ultralytics.com/modes/predict/#inference-arguments
set -e
model=${1:-'yolov8x-worldv2.pt'}
input_img=${2:-$(pwd)/tmp/person.png}
img_name=$(basename $input_img .png)
output_dir=${3:-$(pwd)/tmp/$img_name}

model_name=$(basename $model)

yolo detect predict model=$model \
    source=$input_img \
    project=$output_dir \
    name=$model_name \
    retina_masks=True \
    save_txt=True

#!/bin/bash
# refer: https://docs.ultralytics.com/modes/export/#arguments
set -e
model=${1:-'yolov8x-worldv2.pt'}
dynamic_dim=${2:-True}
contain_nms=${3:-True}
yaml_cfg=${4:-'coco8.yaml'}

dir_name=$(dirname $model)
dynamic_flag=$([ "$dynamic_dim" == "True" ] && echo "dynamic" || echo "static")
nms_flag=$([ "$contain_nms" == "True" ] && echo "with_nms" || echo "no_nms")
onnx_name=$(basename $model .pt)_${dynamic_flag}_${nms_flag}.onnx
# rm $dir_name/*.onnx

yolo export model=$model \
    dynamic=$dynamic_dim \
    nms=$contain_nms \
    format=onnx \
    simplify=True \
    opset=17 \
    optimize=True \
    data=$yaml_cfg

# mv $dir_name/*.onnx $(pwd)/onnx/$onnx_name

#!/bin/bash
# refer: https://docs.ultralytics.com/modes/export/#arguments
set -e

# List of model files and their corresponding data files to convert to ONNX format
declare -A model_data_list=(
    ["./pths/fastsam-x.pt"]="coco8.yaml"
    ["./pths/rtdetr-l.pt"]="coco8.yaml"
    ["./pths/yolo11n.pt"]="coco8.yaml"
    ["./pths/yolo11x.pt"]="coco8.yaml"
    ["./pths/yolo11x-seg.pt"]="coco8.yaml"
    ["./pths/yolo11x-pose.pt"]="coco8.yaml"
    ["./pths/yolov8x-worldv2.pt"]="coco8.yaml"
)

# Function to convert a model to ONNX format
convert_to_onnx() {
    local model=${1}
    local data=${2}
    local dynamic_shape=${3:-True}
    local contain_nms=${4:-True}
    local opset_version=${5:-17}

    local dir_name=$(dirname $model)
    local dynamic_flag=$([ "$dynamic_shape" == "True" ] && echo "dynamic" || echo "static")
    local nms_flag=$([ "$contain_nms" == "True" ] && echo "_nms" || echo "")
    local dataset=$(basename $data .yaml)
    local onnx_name=$(basename $model .pt)_${dynamic_flag}${nms_flag}_${dataset}_ops${opset_version}.onnx
    rm -rf $dir_name/*.onnx

    echo "Converting $model to ONNX format with data file $data..."
    yolo export model=$model \
        dynamic=$dynamic_shape \
        nms=$contain_nms \
        opset=$opset_version \
        data=$data \
        format=onnx \
        simplify=True \
        optimize=True \
        agnostic_nms=True \
        max_det=100

    mv $dir_name/*.onnx $(pwd)/onnx/$onnx_name
    echo "Conversion of $model completed."
}

# Create the ONNX output directory if it doesn't exist
rm -rf onnx && mkdir -p onnx

# Loop through each model file and convert to ONNX format
for model in "${!model_data_list[@]}"; do
    data=${model_data_list[$model]}
    convert_to_onnx "$model" "$data"
done

echo "All models have been converted to ONNX format."

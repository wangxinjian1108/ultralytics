#!/bin/bash
# SAM模型处理图像目录并生成视频

set -e
input_dir=${1:-"./data/images"}
output_dir=${2:-"./tmp/sam_output"}
model=${3:-"sam2.1_b.pt"}
fps=${4:-30}
prompt_points=${5:-""}

# 显示使用帮助信息
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo "使用方法: $0 [输入目录] [输出目录] [模型路径] [帧率] [提示点]"
  echo ""
  echo "参数说明:"
  echo "  输入目录    - 包含图像的目录，默认: ./data/images"
  echo "  输出目录    - 保存处理结果的目录，默认: ./tmp/sam_output"
  echo "  模型路径    - SAM模型路径，默认: sam2.1_b.pt"
  echo "  帧率        - 生成视频的帧率，默认: 30"
  echo "  提示点      - 分割提示点，格式: x1,y1,x2,y2,..."
  echo ""
  echo "示例: $0 ./my_images ./results sam2.1_b.pt 24 \"100,100,200,200\""
  exit 0
fi

# 创建输出目录
mkdir -p "$output_dir"
rm -rf "$output_dir"

# 打印参数信息
echo "处理图像..."
echo "- 输入目录: $input_dir"
echo "- 输出目录: $output_dir"
echo "- 模型: $model"
echo "- 帧率: $fps"
if [ -n "$prompt_points" ]; then
  echo "- 提示点: $prompt_points"
fi

# 检查是否提供了提示点参数
if [ -n "$prompt_points" ]; then
  python scripts/sam_dir.py \
    --model "$model" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --fps "$fps" \
    --prompt-points "$prompt_points"
else
  python scripts/sam_dir.py \
    --model "$model" \
    --input-dir "$input_dir" \
    --output-dir "$output_dir" \
    --fps "$fps"
fi

echo "SAM处理完成，输出保存到 $output_dir" 
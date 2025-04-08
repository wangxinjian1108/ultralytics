from ultralytics import YOLO
import argparse
import cv2
import os
import sys
import glob
import shutil
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random

# Define a fixed color palette for classes
# Using distinct colors that are visually distinguishable
COLOR_PALETTE = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Dark Red
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Blue
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (255, 128, 0),  # Orange
    (255, 0, 128),  # Pink
    (128, 255, 0),  # Lime
    (0, 255, 128),  # Spring Green
]

def get_class_colors(num_classes, class_names=None):
    """
    Generate a consistent color mapping for classes.
    
    Args:
        num_classes: Number of classes to generate colors for
        class_names: Optional list of class names
        
    Returns:
        Dictionary mapping class IDs to colors
    """
    # If we have more classes than colors, extend the palette
    if num_classes > len(COLOR_PALETTE):
        # Generate additional colors by mixing existing ones
        additional_colors = []
        for i in range(num_classes - len(COLOR_PALETTE)):
            # Mix two random colors from the palette
            color1 = random.choice(COLOR_PALETTE)
            color2 = random.choice(COLOR_PALETTE)
            mixed_color = (
                (color1[0] + color2[0]) // 2,
                (color1[1] + color2[1]) // 2,
                (color1[2] + color2[2]) // 2
            )
            additional_colors.append(mixed_color)
        all_colors = COLOR_PALETTE + additional_colors
    else:
        all_colors = COLOR_PALETTE[:num_classes]
    
    # Create a dictionary mapping class IDs to colors
    color_dict = {i: all_colors[i] for i in range(num_classes)}
    
    # If class names are provided, create a mapping from class names to colors
    if class_names:
        name_to_color = {}
        for i, name in enumerate(class_names):
            if i < num_classes:
                name_to_color[name] = color_dict[i]
        return color_dict, name_to_color
    
    return color_dict

def process_image(model, image_path, output_dir, class_colors=None, class_names=None, imgsz=None):
    """
    Process a single image with the segmentation model.
    
    Args:
        model: YOLO model instance
        image_path: Path to the input image
        output_dir: Directory to save the processed image
        class_colors: Dictionary mapping class IDs to colors
        class_names: List of class names to filter by (only these will be shown)
        imgsz: Image size as single int or [height, width]
    """
    try:
        # 确保输入图像文件存在并可读取
        if not os.path.isfile(image_path):
            print(f"错误: 图像文件不存在: {image_path}")
            return False
            
        # 先尝试读取图像，确保它是有效的
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图像: {image_path}")
            return False
            
        # 检查图像尺寸是否有效
        if img.shape[0] <= 0 or img.shape[1] <= 0:
            print(f"错误: 图像尺寸无效: {image_path}, 形状: {img.shape}")
            return False
            
        # 提取文件名，用于输出
        filename = os.path.basename(image_path)
        base_filename = os.path.splitext(filename)[0]
        
        # Run prediction
        results = model(image_path, imgsz=imgsz)
        
        # Process results
        for result in results:
            # Get the original image
            img = result.orig_img.copy()
            
            # 存储检测结果
            detection_results = []
            
            # Draw masks with consistent colors
            if hasattr(result, 'masks') and result.masks is not None:
                for i, mask in enumerate(result.masks):
                    # Get class ID for this instance
                    class_id = int(result.boxes.cls[i].item()) if hasattr(result.boxes, 'cls') else 0
                    
                    # Get class name from model
                    model_class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
                    
                    # 仅处理指定的类别
                    if class_names and model_class_name not in class_names:
                        continue
                    
                    # 如果指定了类别颜色映射，使用它；否则，使用固定颜色
                    if isinstance(class_colors, tuple) and len(class_colors) == 2:
                        # 如果返回了(class_colors, name_to_color)元组
                        _, name_to_color = class_colors
                        color = name_to_color.get(model_class_name, (255, 255, 255))
                    elif isinstance(class_colors, dict):
                        color = class_colors.get(class_id, (255, 255, 255))
                    else:
                        color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]
                    
                    # 转换mask为numpy数组
                    binary_mask = mask.data.cpu().numpy()
                    
                    # 检查并移除多余的维度(如果形状是(1,H,W)，需要转换为(H,W))
                    if len(binary_mask.shape) == 3 and binary_mask.shape[0] == 1:
                        binary_mask = binary_mask.squeeze(0)  # 移除第一个维度
                    
                    # 转换为uint8类型
                    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
                    
                    # 确保mask尺寸与图像一致
                    if binary_mask.shape != img.shape[:2]:
                        try:
                            # 确保图像尺寸大于0
                            if img.shape[1] > 0 and img.shape[0] > 0:
                                binary_mask = cv2.resize(
                                    binary_mask, 
                                    (img.shape[1], img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            else:
                                print(f"警告: 图像 {image_path} 尺寸无效: {img.shape}")
                                continue
                        except Exception as e:
                            print(f"警告: 无法调整掩码大小，图像 {image_path}, 形状 {img.shape}, 掩码形状 {binary_mask.shape}: {str(e)}")
                            continue
                    
                    # 将掩码转换为轮廓多边形（用于JSON输出）
                    try:
                        # 确保二值掩码是有效的（非空且值为0或1）
                        if np.any(binary_mask > 0):
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        else:
                            print(f"警告: 图像 {image_path} 的掩码为空")
                            contours = []
                    except Exception as e:
                        print(f"警告: 在图像 {image_path} 上查找轮廓时出错: {str(e)}")
                        contours = []
                    
                    # 简化并存储轮廓点
                    mask_polygons = []
                    for contour in contours:
                        try:
                            # 确保轮廓有足够的点
                            if len(contour) >= 3:
                                # 简化轮廓，减少点数
                                epsilon = 0.005 * cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, epsilon, True)
                                if len(approx) > 2:  # 确保至少有3个点形成一个多边形
                                    # 将轮廓转换为点列表 [[x1,y1], [x2,y2], ...]
                                    polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
                                    mask_polygons.append(polygon)
                        except Exception as e:
                            print(f"警告: 在图像 {image_path} 上处理轮廓时出错: {str(e)}")
                            continue
                    
                    # 获取边界框
                    if len(contours) > 0:
                        try:
                            bbox = result.boxes.xyxy[i].cpu().numpy().astype(np.int32)
                            x1, y1, x2, y2 = bbox
                            
                            # 确保边界框坐标有效
                            if x1 < 0: x1 = 0
                            if y1 < 0: y1 = 0
                            if x2 >= img.shape[1]: x2 = img.shape[1] - 1
                            if y2 >= img.shape[0]: y2 = img.shape[0] - 1
                            
                            # 确保边界框有效（宽度和高度都大于0）
                            if x2 > x1 and y2 > y1:
                                conf = float(result.boxes.conf[i].item())
                                
                                # 添加到检测结果
                                detection_results.append({
                                    'type': model_class_name,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'mask_polygons': mask_polygons
                                })
                            else:
                                print(f"警告: 图像 {image_path} 的第 {i} 个检测结果边界框无效: [{x1}, {y1}, {x2}, {y2}]")
                        except Exception as e:
                            print(f"警告: 在图像 {image_path} 上处理边界框时出错: {str(e)}")
                            continue
                    
                    # Apply color to mask
                    colored_mask = np.zeros_like(img)
                    colored_mask[binary_mask > 0] = color
                    
                    # Blend with original image
                    alpha = 0.5
                    img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
                    
                    # Draw contour
                    cv2.drawContours(img, contours, -1, color, 2)
                    
                    # Add class name label
                    if len(contours) > 0:
                        # Find the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Add text background
                        text_size = cv2.getTextSize(model_class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(img, (x, y - 25), (x + text_size[0], y), color, -1)
                        
                        # Add text
                        cv2.putText(img, model_class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 保存检测结果为JSON
            if detection_results:
                json_path = os.path.join(output_dir, f"{base_filename}_detections.json")
                full_results = {
                    "image": {
                        "path": image_path,
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "channels": img.shape[2] if len(img.shape) > 2 else 1
                    },
                    "detections": detection_results
                }
                with open(json_path, 'w') as f:
                    json.dump(full_results, f, indent=2)
            
            # Save the annotated image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)
            
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_video_ffmpeg(image_dir, output_video_path, fps=30):
    """
    Create a video from a directory of images using ffmpeg.
    
    Args:
        image_dir: Directory containing the processed images
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    try:
        # Get list of image files
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print("No images found in the directory")
            return False
        
        # Create a temporary directory for sequential frames
        temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy and rename frames sequentially for ffmpeg
        print("Preparing frames for video...")
        for i, img_file in enumerate(tqdm(image_files, desc="Preparing frames")):
            img_path = os.path.join(image_dir, img_file)
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(frame_path, img)
        
        # Ensure output path has .mp4 extension
        video_path = os.path.splitext(output_video_path)[0] + '.mp4'
        
        # Use ffmpeg to create the video
        print("Creating video with ffmpeg...")
        ffmpeg_cmd = f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
        os.system(ffmpeg_cmd)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"Successfully created video: {video_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        # Clean up temporary directory if it exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def process_directory(model_path, input_dir, output_dir, fps=30, class_names=None, imgsz=None):
    """
    Process all images in a directory and create a video.
    
    Args:
        model_path: Path to the YOLO model file
        input_dir: Directory containing input images
        output_dir: Directory to save processed images and video
        fps: Frames per second for the output video
        class_names: Optional list of class names to filter by (only these will be shown)
        imgsz: Image size as single int or [height, width]
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the model
        model = YOLO(model_path, task='segment')
        
        # Get list of image files
        image_files = sorted([f for f in os.listdir(input_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return False
            
        # 检查第一个图像文件是否有效
        first_image = image_files[0]
        first_image_path = os.path.join(input_dir, first_image)
        try:
            test_img = cv2.imread(first_image_path)
            if test_img is None or test_img.size == 0:
                print(f"无法读取第一个图像文件: {first_image_path}")
                return False
                
            # 确保图像尺寸有效
            if test_img.shape[0] <= 0 or test_img.shape[1] <= 0:
                print(f"图像尺寸无效: {first_image_path}, 形状: {test_img.shape}")
                return False
                
            # 尝试在第一个图像上运行模型
            first_result = model(first_image_path, imgsz=imgsz)
            
            # 如果提供了imgsz，显示正在使用的输入分辨率
            if imgsz:
                print(f"使用输入分辨率: {imgsz}")
                
            # 保存模型类别名称
            model_names_path = os.path.join(output_dir, "model_class_names.json")
            with open(model_names_path, 'w') as f:
                json.dump(first_result[0].names, f, indent=4)
                print(f"保存模型类别名称到 {model_names_path}")
                
        except Exception as e:
            print(f"处理第一个图像时出错: {first_image_path}")
            print(f"错误: {str(e)}")
            traceback.print_exc()
            return False
        
        # Generate consistent color mapping for classes
        # 为指定的类别创建颜色映射
        predefined_colors = {
            'person': (0, 0, 255),       # 红色
            'car': (0, 255, 0),          # 绿色
            'bicycle': (255, 0, 0),      # 蓝色
            'motorcycle': (255, 255, 0), # 青色
            'truck': (255, 0, 255),      # 紫色
            'bus': (0, 255, 255),        # 黄色
            'traffic cone': (128, 0, 128),  # 紫色
            'traffic light': (255, 128, 0),  # 橙色
            'stop sign': (0, 128, 255),     # 橙红色
        }
        
        name_to_color = {}
        if class_names:
            for name in class_names:
                if name in predefined_colors:
                    name_to_color[name] = predefined_colors[name]
                else:
                    # 对于未预定义的类别，使用深灰色
                    name_to_color[name] = (80, 80, 80)
            print(f"处理指定的 {len(class_names)} 个类别: {class_names}")
        else:
            # 使用模型的所有类别
            model_class_names = first_result[0].names
            for class_id, name in model_class_names.items():
                if name in predefined_colors:
                    name_to_color[name] = predefined_colors[name]
                else:
                    name_to_color[name] = COLOR_PALETTE[int(class_id) % len(COLOR_PALETTE)]
            print(f"处理模型的所有 {len(model_class_names)} 个类别")
        
        # Process each image with consistent colors
        print("Processing images...")
        failed_images = []
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            success = process_image(model, image_path, output_dir, (None, name_to_color), class_names, imgsz)
            if not success:
                failed_images.append(image_file)
        
        # 报告处理结果
        total_images = len(image_files)
        processed_images = total_images - len(failed_images)
        print(f"处理完成: {processed_images}/{total_images} 图像成功处理")
        if failed_images:
            print(f"无法处理的图像 ({len(failed_images)}):")
            for img in failed_images[:min(10, len(failed_images))]:
                print(f"  - {img}")
            if len(failed_images) > 10:
                print(f"  ... 以及另外 {len(failed_images) - 10} 个文件")
        
        # Create video from processed images
        print("Creating video...")
        video_path = os.path.join(output_dir, "segmentation_video")
        if create_video_ffmpeg(output_dir, video_path, fps):
            print(f"Video creation completed")
        else:
            print("Failed to create video")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a directory of images with YOLO segmentation and create a video')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save processed images and video')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video (default: 30)')
    parser.add_argument('--class-names', type=str, help='JSON file containing list of class names or comma-separated list of class names')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[544, 960], help='Image size as single int or [height, width], e.g. --imgsz 640 or --imgsz 640 640')
    
    args = parser.parse_args()
    
    # 转换imgsz参数，确保格式正确
    imgsz = args.imgsz
    if len(imgsz) == 1:
        # 如果只提供一个数字，则使用正方形尺寸
        imgsz = imgsz[0]
    elif len(imgsz) > 2:
        # 如果提供超过两个数字，只使用前两个
        imgsz = imgsz[:2]
    print(f"图像尺寸设置为: {imgsz}")
    
    # Parse class names
    class_names = None
    if args.class_names:
        if os.path.isfile(args.class_names):
            # Load from JSON file
            try:
                with open(args.class_names, 'r') as f:
                    class_names = json.load(f)
                print(f"Loaded {len(class_names)} class names from {args.class_names}")
            except Exception as e:
                print(f"Error loading class names from file: {str(e)}")
                print("Using default class names instead")
        else:
            # Parse dot-separated list and replace underscores with spaces
            class_names = [name.strip() for name in args.class_names.split('.')]
            # Replace underscores with spaces
            class_names = [name.replace('_', ' ') for name in class_names]
            print(f"Using {len(class_names)} class names: {class_names}")
    
    success = process_directory(args.model, args.input_dir, args.output_dir, args.fps, class_names, imgsz)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 
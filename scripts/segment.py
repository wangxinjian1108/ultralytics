from ultralytics import YOLO
import argparse
import cv2
import os
import sys
import json
import numpy as np
from pathlib import Path

def load_class_names(class_names_arg):
    """
    Load class names from either a JSON file or a comma-separated string.
    
    Args:
        class_names_arg (str): Path to JSON file or comma-separated class names
        
    Returns:
        list: List of class names
    """
    if not class_names_arg:
        return None
        
    # Check if the argument is a path to a JSON file
    if os.path.isfile(class_names_arg):
        try:
            with open(class_names_arg, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'names' in data:
                    return data['names']
                else:
                    print(f"Warning: JSON file {class_names_arg} does not contain a list or a dict with 'names' key")
                    return None
        except Exception as e:
            print(f"Error loading class names from {class_names_arg}: {str(e)}")
            return None
    else:
        # Treat as comma-separated class names
        names = [name.strip() for name in class_names_arg.split('.')]
        # replace "_" by " "
        names = [name.replace('_', ' ') for name in names]
        return names

def run_segmentation(model_path, image_path, save_path=None, class_names=None, imgsz=None):
    """
    Run YOLO segmentation on an image and visualize the results.
    
    Args:
        model_path (str): Path to the YOLO model file
        image_path (str): Path to the input image or URL
        save_path (str, optional): Path to save the visualized image. If None, will show the image.
        class_names (list, optional): List of class names to use for visualization
        imgsz (int or list, optional): Image size as single int or [height, width]
    
    Returns:
        tuple: (results, detection_json_path) - The model results and path to the saved JSON file
    """
    try:
        # Create a save directory for results if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load the model with explicit task specification and overrides
        model = YOLO(model_path, task='segment')
        
        # Run prediction with custom image size if specified
        results = model.predict(source=image_path, save=False, imgsz=imgsz)
        
        # output the result names to json
        with open('result_names.json', 'w') as f:
            json.dump(results[0].names, f, indent=4)
            print(f"Saved result names to result_names.json")
            
        if imgsz:
            print(f"Using input resolution: {imgsz}")
        
        # Process and visualize results manually
        for result in results:
            # Get the original image
            orig_img = result.orig_img.copy()
            
            # Get all the masks, boxes, and class information
            masks = result.masks.data if hasattr(result, 'masks') and result.masks is not None else None
            boxes = result.boxes.data if hasattr(result, 'boxes') and result.boxes is not None else None
            
            if masks is None or boxes is None:
                print("No masks or boxes detected in the image.")
                if save_path:
                    cv2.imwrite(save_path, orig_img)
                else:
                    cv2.imshow("Original Image", orig_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                continue
            
            # Get model class names (all classes the model knows)
            model_class_names = result.names
            
            # Create fixed color mapping for classes (不使用随机颜色)
            colors_map = {}
            # 预定义一些鲜明的颜色（BGR格式）
            predefined_colors = {
                'person': [0, 0, 255],     # 红色
                'car': [0, 255, 0],        # 绿色
                'bicycle': [255, 0, 0],    # 蓝色
                'motorcycle': [255, 255, 0],  # 青色
                'truck': [255, 0, 255],    # 紫色
                'bus': [0, 255, 255],      # 黄色
                'traffic cone': [128, 0, 128],  # 紫色
                'traffic light': [255, 128, 0], # 橙色
                'stop sign': [0, 128, 255],   # 橙红色
            }
            
            if class_names:
                for name in class_names:
                    if name in predefined_colors:
                        colors_map[name] = predefined_colors[name]
                    else:
                        # 对于未预定义的类别，使用深灰色
                        colors_map[name] = [80, 80, 80]
            
            # Create a blank mask for all classes (will be filled with class-specific colors)
            composite_mask = np.zeros_like(orig_img)
            
            # 存储用于JSON输出的完整检测结果
            detection_results = []
            
            # Process all detections
            annotations = []
            for i in range(len(boxes)):
                # Get the class ID and confidence
                cls_id = int(boxes[i, 5].item())
                conf = float(boxes[i, 4].item())
                
                # Get the class name
                cls_name = model_class_names[cls_id]
                
                # Skip if not in the class_names list (when class_names is provided)
                if class_names and cls_name not in class_names:
                    continue
                
                # Get the box coordinates
                bbox = boxes[i, :4].cpu().numpy().astype(np.int32)
                x1, y1, x2, y2 = bbox
                
                # Get the corresponding mask
                if masks is not None and i < len(masks):
                    mask = masks[i].cpu().numpy()
                    
                    # 调整掩码大小以匹配原始图像尺寸
                    # 注意：掩码是二维的，需要使用INTER_NEAREST或INTER_LINEAR以保持二值性质
                    if mask.shape != orig_img.shape[:2]:
                        mask = cv2.resize(
                            mask, 
                            (orig_img.shape[1], orig_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    # Convert mask from 0-1 to 0-255
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    
                    # 为JSON编码掩码 - 找到掩码轮廓并将其编码为多边形轮廓点
                    # 这样掩码就可以用紧凑的点列表而不是完整的二值图像来表示
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 如果找到轮廓，简化并存储它们
                    mask_polygons = []
                    for contour in contours:
                        # 简化轮廓，减少点数
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        if len(approx) > 2:  # 确保至少有3个点形成一个多边形
                            # 将轮廓转换为点列表 [[x1,y1], [x2,y2], ...]
                            polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
                            mask_polygons.append(polygon)
                    
                    # 为当前对象存储完整的检测结果
                    detection_results.append({
                        'type': cls_name,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'mask_polygons': mask_polygons
                    })
                    
                    # Assign color based on class
                    if class_names:
                        color = colors_map.get(cls_name, [0, 255, 0])  # Default to green if not found
                    else:
                        # 使用预定义的颜色，如果没有则使用灰色
                        color = predefined_colors.get(cls_name, [80, 80, 80])
                    
                    # Create a colored mask
                    colored_mask = np.zeros_like(orig_img)
                    colored_mask[binary_mask > 0] = color
                    
                    # Add the mask to the composite
                    alpha = 0.5  # Transparency
                    mask_area = (binary_mask > 0)
                    composite_mask[mask_area] = colored_mask[mask_area]
                    
                    # Store annotations for later use
                    annotations.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'color': color
                    })
            
            # Blend the composite mask with the original image
            result_img = cv2.addWeighted(orig_img, 1, composite_mask, 0.5, 0)
            
            # Draw bounding boxes and labels
            for ann in annotations:
                x1, y1, x2, y2 = ann['bbox']
                cls_name = ann['class_name']
                conf = ann['conf']
                color = ann['color']
                
                # Draw rectangle
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label with class name and confidence
                label = f"{cls_name} {conf:.2f}"
                
                # Draw label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    result_img, 
                    (x1, y1 - text_size[1] - 5), 
                    (x1 + text_size[0], y1), 
                    color, 
                    -1
                )
                
                # Draw text (white on colored background)
                cv2.putText(
                    result_img, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
            
            # Add a legend for class names
            if class_names:
                for i, name in enumerate(class_names):
                    color = colors_map.get(name, [0, 255, 0])
                    y_pos = 30 + i * 30
                    cv2.putText(
                        result_img,
                        name,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
            
            if save_path:
                # Save the annotated image
                cv2.imwrite(save_path, result_img)
                print(f"Saved visualization to {save_path}")
                
                # 保存检测结果为JSON文件
                json_save_path = save_path.rsplit('.', 1)[0] + '_detections.json'
                
                # 创建包含图像信息的完整结果对象
                full_results = {
                    "image": {
                        "path": image_path,
                        "width": orig_img.shape[1],
                        "height": orig_img.shape[0],
                        "channels": orig_img.shape[2] if len(orig_img.shape) > 2 else 1
                    },
                    "detections": detection_results
                }
                
                with open(json_save_path, 'w') as f:
                    json.dump(full_results, f, indent=2)
                print(f"Saved detection results to {json_save_path}")
            else:
                # Display the image
                cv2.imshow("Segmentation Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        return results, detection_results
    except Exception as e:
        print(f"Error running segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Make sure you're using a compatible model and the latest version of ultralytics")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run YOLO segmentation on an image')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--source', type=str, required=True, help='Path to the input image or URL')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save the output image')
    parser.add_argument('--class-names', type=str, help='Path to JSON file with class names or comma-separated list of class names')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640], help='Image size as single int or [height, width], e.g. --imgsz 640 or --imgsz 640 640')
    
    args = parser.parse_args()
    
    # 转换imgsz参数，确保格式正确
    imgsz = args.imgsz
    if len(imgsz) == 1:
        # 如果只提供一个数字，则使用正方形尺寸
        imgsz = imgsz[0]
    elif len(imgsz) > 2:
        # 如果提供超过两个数字，只使用前两个
        imgsz = imgsz[:2]
    print(f"Image size set to: {imgsz}")
    
    # Load class names if provided
    class_names = load_class_names(args.class_names)
    if class_names:
        print(f"Using custom class names: {class_names}")
    
    # Create output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get the input filename without extension
    input_filename = Path(args.source).stem
    save_path = str(output_dir / f"{input_filename}_segmented.jpg")
    
    # 运行分割并获取结果
    results, detection_results = run_segmentation(args.model, args.source, save_path, class_names, imgsz)
    
    # 如果没有在run_segmentation中保存JSON（例如，没有指定save_path），这里手动保存
    if save_path and detection_results:
        json_save_path = save_path.rsplit('.', 1)[0] + '_detections.json'
        if not os.path.exists(json_save_path):
            # 获取原图像尺寸信息
            try:
                orig_img = cv2.imread(args.source)
                image_info = {
                    "path": args.source,
                    "width": orig_img.shape[1],
                    "height": orig_img.shape[0],
                    "channels": orig_img.shape[2] if len(orig_img.shape) > 2 else 1
                }
            except:
                # 如果无法读取图像，提供基本信息
                image_info = {
                    "path": args.source,
                    "width": 0,
                    "height": 0,
                    "channels": 3
                }
            
            # 创建完整结果对象
            full_results = {
                "image": image_info,
                "detections": detection_results
            }
            
            with open(json_save_path, 'w') as f:
                json.dump(full_results, f, indent=2)
            print(f"Saved detection results to {json_save_path}")
    
    return results, detection_results

if __name__ == '__main__':
    main()

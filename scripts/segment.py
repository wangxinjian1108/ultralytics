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
        return [name.strip() for name in class_names_arg.split(',')]

def run_segmentation(model_path, image_path, save_path=None, class_names=None, imgsz=None):
    """
    Run YOLO segmentation on an image and visualize the results.
    
    Args:
        model_path (str): Path to the YOLO model file
        image_path (str): Path to the input image or URL
        save_path (str, optional): Path to save the visualized image. If None, will show the image.
        class_names (list, optional): List of class names to use for visualization
        imgsz (int or list, optional): Image size as single int or [height, width]
    """
    try:
        # Create a save directory for results if needed
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load the model with explicit task specification and overrides
        model = YOLO(model_path, task='segment')
        
        # Run prediction with custom image size if specified
        results = model.predict(source=image_path, save=False, imgsz=imgsz)
        
        # Print input resolution being used
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
            
            # Create color mapping for classes
            colors_map = {}
            if class_names:
                num_classes = len(class_names)
                random_colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
                for i, name in enumerate(class_names):
                    colors_map[name] = random_colors[i].tolist()
            
            # Create a blank mask for all classes (will be filled with class-specific colors)
            composite_mask = np.zeros_like(orig_img)
            
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
                    
                    # Assign color based on class
                    if class_names:
                        color = colors_map.get(cls_name, [0, 255, 0])  # Default to green if not found
                    else:
                        # Generate a color based on class ID
                        color = (int(np.random.randint(0, 255)), 
                                 int(np.random.randint(0, 255)), 
                                 int(np.random.randint(0, 255)))
                    
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
            else:
                # Display the image
                cv2.imshow("Segmentation Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        return results
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
    parser.add_argument('--imgsz', type=int, nargs='+', default=None, help='Image size as single int or [height, width], e.g. --imgsz 640 or --imgsz 640 640')
    
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
    
    results = run_segmentation(args.model, args.source, save_path, class_names, imgsz)
    return results

if __name__ == '__main__':
    main()

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

def process_image(model, image_path, output_dir, class_colors=None, class_names=None):
    """
    Process a single image with the segmentation model.
    
    Args:
        model: YOLO model instance
        image_path: Path to the input image
        output_dir: Directory to save the processed image
        class_colors: Dictionary mapping class IDs to colors
        class_names: List of class names
    """
    try:
        # Run prediction
        results = model(image_path)
        
        # Process results
        for result in results:
            # Get the original image
            img = result.orig_img.copy()
            
            # Draw masks with consistent colors
            if hasattr(result, 'masks') and result.masks is not None:
                for i, mask in enumerate(result.masks):
                    # Get class ID for this instance
                    class_id = int(result.boxes.cls[i].item()) if hasattr(result.boxes, 'cls') else 0
                    
                    # Get class name if available
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                    
                    # Get color for this class
                    color = class_colors.get(class_id, (255, 255, 255)) if class_colors else (255, 255, 255)
                    
                    # Convert mask to binary
                    binary_mask = mask.data.cpu().numpy().astype(np.uint8)
                    
                    # Apply color to mask
                    colored_mask = np.zeros_like(img)
                    colored_mask[binary_mask > 0] = color
                    
                    # Blend with original image
                    alpha = 0.5
                    img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
                    
                    # Draw contour
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, color, 2)
                    
                    # Add class name label
                    if len(contours) > 0:
                        # Find the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Add text background
                        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(img, (x, y - 25), (x + text_size[0], y), color, -1)
                        
                        # Add text
                        cv2.putText(img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Save the annotated image
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, img)
            
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
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

def process_directory(model_path, input_dir, output_dir, fps=30, class_names=None):
    """
    Process all images in a directory and create a video.
    
    Args:
        model_path: Path to the YOLO model file
        input_dir: Directory containing input images
        output_dir: Directory to save processed images and video
        fps: Frames per second for the output video
        class_names: Optional list of class names
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
        
        # First pass: determine number of classes
        print("Analyzing classes in images...")
        class_ids = set()
        for image_file in tqdm(image_files[:min(10, len(image_files))], desc="Analyzing classes"):
            image_path = os.path.join(input_dir, image_file)
            results = model(image_path)
            for result in results:
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'cls'):
                    for cls in result.boxes.cls:
                        class_ids.add(int(cls.item()))
        
        # Generate consistent color mapping for classes
        num_classes = len(class_ids) if class_ids else 1
        
        # If class names are provided, use them
        if class_names:
            # If we have more class names than detected classes, use all class names
            if len(class_names) > num_classes:
                num_classes = len(class_names)
                print(f"Using provided class names: {class_names}")
            else:
                print(f"Using provided class names for {len(class_names)} classes")
        else:
            print(f"Using generic class names for {num_classes} classes")
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Generate color mapping
        class_colors = get_class_colors(num_classes, class_names)
        if isinstance(class_colors, tuple):
            class_colors, name_to_color = class_colors
            print("Class color mapping:")
            for name, color in name_to_color.items():
                print(f"  {name}: RGB{color}")
        
        # Process each image with consistent colors
        print("Processing images...")
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            process_image(model, image_path, output_dir, class_colors, class_names)
        
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
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a directory of images with YOLO segmentation and create a video')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save processed images and video')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video (default: 30)')
    parser.add_argument('--class-names', type=str, help='JSON file containing list of class names or comma-separated list of class names')
    
    args = parser.parse_args()
    
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
            # Parse comma-separated list
            class_names = [name.strip() for name in args.class_names.split(',')]
            print(f"Using {len(class_names)} class names: {class_names}")
    
    success = process_directory(args.model, args.input_dir, args.output_dir, args.fps, class_names)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 
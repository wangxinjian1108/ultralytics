import os
import random
import shutil
import argparse
import logging
from collections import defaultdict

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract image and label pairs for each class
def extract_images_and_labels_for_class(class_id, img_dir, label_dir, n_images_per_class):
    selected_images = []

    logging.info(f"Processing class {class_id}...")

    # Loop through all images in the directory
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))  # Assuming .jpg images

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Check if the class_id is present in the annotation
                    if int(line.split()[0]) == class_id:
                        selected_images.append((img_path, label_path))
                        break

    # Select up to n_images_per_class images
    selected_images = random.sample(selected_images, min(n_images_per_class, len(selected_images)))

    logging.info(f"Found {len(selected_images)} images for class {class_id}.")
    return selected_images

# Function to convert label file to YOLO format
def convert_to_yolo_format(img_path, label_path, output_label_dir, class_id, img_width, img_height):
    label_out_path = os.path.join(output_label_dir, os.path.basename(label_path))

    with open(label_path, 'r') as f:
        lines = f.readlines()

    with open(label_out_path, 'w') as f_out:
        for line in lines:
            parts = line.strip().split()
            c_id = int(parts[0])  # Original class ID
            if c_id == class_id:  # Only write labels for the desired class
                x_center = (float(parts[1]) + float(parts[3])) / 2 / img_width
                y_center = (float(parts[2]) + float(parts[4])) / 2 / img_height
                width = (float(parts[3]) - float(parts[1])) / img_width
                height = (float(parts[4]) - float(parts[2])) / img_height

                # Write in YOLO format
                f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Main function to create the tiny dataset
def create_tiny_dataset(args):
    # Convert selected_classes from comma-separated string to list of integers
    selected_classes = [int(x) for x in args.selected_classes.split(',')]

    logging.info(f"Starting extraction for {len(selected_classes)} classes: {selected_classes}")

    # Directories for images and annotations
    train_img_dir = os.path.join(args.dataset_path, 'images/train')
    train_label_dir = os.path.join(args.dataset_path, 'labels/train')

    # Number of images to extract per class
    n_images_per_class = args.total_images // len(selected_classes)  # Total images, balanced across classes

    # Create output directories for the tiny dataset
    os.makedirs(os.path.join(args.output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels/train'), exist_ok=True)

    # Dictionary to store the selected images per class
    selected_images_by_class = defaultdict(list)

    # Extract images for each selected class and store them in the dictionary
    for class_id in selected_classes:
        logging.info(f"Extracting images for class {class_id}...")
        selected_images = extract_images_and_labels_for_class(class_id, train_img_dir, train_label_dir, n_images_per_class)
        selected_images_by_class[class_id].extend(selected_images)

    # Ensure that the total number of images selected is the desired number (or less if there aren't enough images)
    selected_images = []
    for class_id, images in selected_images_by_class.items():
        selected_images.extend(images)

    logging.info(f"Total selected images before shuffling: {len(selected_images)}")

    # Shuffle the selected images to ensure random distribution
    random.shuffle(selected_images)

    # Copy the selected images and labels to the output directory and convert labels to YOLO format
    for img_path, label_path in selected_images:
        # Copy the image
        shutil.copy(img_path, os.path.join(args.output_dir, 'images/train', os.path.basename(img_path)))

        # Get image size to adjust label format
        from PIL import Image
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # Convert label to YOLO format
        convert_to_yolo_format(img_path, label_path, os.path.join(args.output_dir, 'labels/train'), class_id, img_width, img_height)

    logging.info(f"Successfully extracted {len(selected_images)} images for the tiny dataset.")

    # Create the data.yaml file for Ultralytics YOLO
    data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as yaml_file:
        yaml_file.write(f"train: {os.path.join(args.output_dir, 'images/train')}\n")
        yaml_file.write(f"val: {os.path.join(args.output_dir, 'images/train')}\n")  # You can adjust this for a separate validation set
        yaml_file.write(f"nc: {len(selected_classes)}\n")
        yaml_file.write("names:\n")
        for i, class_id in enumerate(selected_classes):
            yaml_file.write(f"  {i}: Class{class_id}\n")

    logging.info(f"Dataset YAML file created: {data_yaml_path}")

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Extract a balanced subset of images from the Object365 dataset.")
    
    # Adding arguments for dataset path, output directory, and selected classes
    parser.add_argument('--dataset-path', required=True, type=str, help="Path to the root directory of the Object365 dataset")
    parser.add_argument('--output-dir', required=True, type=str, help="Path to store the tiny dataset")
    parser.add_argument('--selected-classes', required=True, type=str, help="Comma-separated list of class indices to select")
    parser.add_argument('--total-images', default=10000, type=int, help="Total number of images to extract (default 10000)")
    
    return parser.parse_args()

# Entry point of the script
if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    
    # Create the tiny dataset
    create_tiny_dataset(args)

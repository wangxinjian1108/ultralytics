from ultralytics import YOLO
import argparse
import cv2
import os
import sys

def run_segmentation(model_path, image_path, save_path=None):
    """
    Run YOLO segmentation on an image and visualize the results.
    
    Args:
        model_path (str): Path to the YOLO model file
        image_path (str): Path to the input image or URL
        save_path (str, optional): Path to save the visualized image. If None, will show the image.
    """
    try:
        # Load the model with explicit task specification
        model = YOLO(model_path, task='segment')
        
        # Run prediction
        results = model(image_path)
        
        # Process and visualize results
        for result in results:
            # Get the original image
            orig_img = result.orig_img
            
            # Draw the segmentation masks on the image
            annotated_img = result.plot()
            
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the annotated image
                cv2.imwrite(save_path, annotated_img)
                print(f"Saved visualization to {save_path}")
            else:
                # Display the image
                cv2.imshow("Segmentation Result", annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        return results
    except Exception as e:
        print(f"Error running segmentation: {str(e)}")
        print("Make sure you're using a compatible model and the latest version of ultralytics")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run YOLO segmentation on an image')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image or URL')
    parser.add_argument('--save', type=str, help='Path to save the visualized image (optional)')
    
    args = parser.parse_args()
    
    results = run_segmentation(args.model, args.image, args.save)
    return results

if __name__ == '__main__':
    main()

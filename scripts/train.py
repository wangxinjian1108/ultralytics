from ultralytics import YOLO
import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train a YOLO model on a custom dataset')
    parser.add_argument('--model', type=str, default='yolo12n.pt', help='Model file path (.pt or .yaml)')
    parser.add_argument('--data', type=str, default='coco8.yaml', help='Dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto-select)')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--name', type=str, default='exp', help='Name of the experiment')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--inference', type=str, default='', help='Run inference after training on this image/video')
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.environ["MASTER_PORT"] = "38200"
    # Load a pretrained YOLO model
    model = YOLO(args.model)

    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        val=False,
    )
    
    # Run inference if specified
    if args.inference and os.path.exists(args.inference):
        results = model(args.inference)
        print(f"Inference results saved to {Path(args.project) / args.name}")

if __name__ == "__main__":
    main()
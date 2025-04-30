# Ultralytics YOLO Training Script

This directory contains scripts for training Ultralytics YOLO models with custom datasets.

## Training with a Custom Dataset

The `train.py` script allows you to train a YOLO model on any custom dataset, including large datasets like Object365.

### Prerequisites

1. Install Ultralytics:
   ```bash
   pip install ultralytics
   ```

2. Prepare your dataset in YOLO format:
   - Images in a directory structure
   - Annotation text files (.txt) with the same name as the images
   - A YAML configuration file defining dataset paths and classes

### Example: Training with Object365 Dataset

1. **Option A**: Manually download and prepare the Object365 dataset
   - Download from the [Objects365 website](https://www.objects365.org/)
   - Update the `object365.yaml` file with the correct paths to your dataset

2. **Option B**: Automatically download the dataset when first used
   - The dataset will be downloaded to the path specified in the YAML file
   - To customize the download location, set the `OBJECTS365_DIR` environment variable:
     ```bash
     # Download to a custom directory
     export OBJECTS365_DIR=/path/to/your/custom/directory
     python train.py --data object365.yaml
     
     # Or specify in one command
     OBJECTS365_DIR=/path/to/your/custom/directory python train.py --data object365.yaml
     ```
   - Note: The full Objects365 dataset is very large (~712GB). Make sure you have sufficient disk space.

3. Run the training script:
   ```bash
   python train.py --model yolo12n.pt --data object365.yaml --epochs 100 --batch 16
   ```

### Command-line Arguments

- `--model`: Path to a pretrained model (.pt) or model configuration (.yaml) (default: "yolo12n.pt")
- `--data`: Path to dataset configuration file (.yaml) (default: "coco8.yaml")
- `--epochs`: Number of training epochs (default: 100)
- `--imgsz`: Input image size (default: 640)
- `--batch`: Batch size (default: 16)
- `--device`: Device to use ('0', '0,1,2,3', 'cpu', etc.) (default: auto-select)
- `--workers`: Number of dataloader workers (default: 8)
- `--name`: Experiment name (default: 'exp')
- `--project`: Project directory (default: 'runs/train')
- `--inference`: Run inference on this image/video after training (default: none)

### Example: Custom Dataset Training

1. Create a YAML configuration file for your dataset (see `object365.yaml` as an example)
2. Run the training script with your configuration:
   ```bash
   python train.py --model yolo12n.pt --data your_dataset.yaml --epochs 50 --batch 8
   ```

3. After training, the model will be saved in `runs/train/exp/weights/best.pt` (or the name specified with `--name`)

### Running Inference

After training, you can run inference on an image or video:

```bash
python train.py --model runs/train/exp/weights/best.pt --inference path/to/image.jpg
```

Or specify both training and inference in one command:

```bash
python train.py --model yolo12n.pt --data your_dataset.yaml --epochs 50 --inference path/to/image.jpg
``` 
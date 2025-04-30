spark-submit py_scripts/dataset_ops/yolo_subset_extract_and_split.py \
    --categories 0 5 34 46 49 55 58 65 66 \
    --remapped_categories 0 1 2 3 4 5 6 7 8 \
    --output_dir /mnt/juicefs/xinjian/open_dataset/object365_tiny50000 \
    --original_dir /mnt/juicefs/xinjian/open_dataset/object365 \
    --min_count 50000

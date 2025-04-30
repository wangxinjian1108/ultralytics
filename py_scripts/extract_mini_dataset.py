import argparse
import json
import os
import shutil
import random
from pyspark.sql import SparkSession
from typing import Set, Dict, List

def main():
    parser = argparse.ArgumentParser(description="Extract subset from Object365 dataset")
    parser.add_argument('--categories', type=int, nargs='+', required=True, help='Target category IDs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--original_dir', type=str, required=True, help='Original dataset directory')
    parser.add_argument('--min_count', type=int, required=True, help='Minimum number of images per category')
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Object365SubsetExtractor").getOrCreate()
    sc = spark.sparkContext

    # 加载标签数据
    annotations_path = os.path.join(args.original_dir, 'annotations', 'annotations.json')
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    # 创建Spark RDD并处理
    target_categories = set(args.categories)
    annotations_rdd = sc.parallelize(annotations)

    # 过滤目标类别并提取(category_id, image_id)
    filtered_rdd = annotations_rdd.filter(lambda ann: ann['category_id'] in target_categories)\
                                  .map(lambda ann: (ann['category_id'], ann['image_id']))

    # 聚合每个类别的唯一image_id
    def seq_op(acc: Set[int], image_id: int):
        acc.add(image_id)
        return acc

    def comb_op(acc1: Set[int], acc2: Set[int]):
        return acc1.union(acc2)

    category_images = filtered_rdd.aggregateByKey(set(), seq_op, comb_op)

    # 选择满足最小数量的image_id
    selected_images_rdd = category_images.flatMap(lambda x: random.sample(list(x[1]), min(args.min_count, len(x[1]))) if len(x[1]) > 0 else [])

    # 收集并去重
    all_selected_images = selected_images_rdd.distinct().collect()

    # 准备输出数据
    output_images = [images[img_id] for img_id in all_selected_images if img_id in images]
    output_annotations = [ann for ann in annotations if ann['image_id'] in all_selected_images]

    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    # 保存新的annotations文件
    output_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": output_images,
        "annotations": output_annotations
    }
    with open(os.path.join(args.output_dir, 'annotations', 'annotations.json'), 'w') as f:
        json.dump(output_data, f)

    # 并行复制图像文件
    images_rdd = sc.parallelize(output_images)
    def copy_image(img):
        src = os.path.join(args.original_dir, 'images', img['file_name'])
        dst = os.path.join(args.output_dir, 'images', img['file_name'])
        if os.path.exists(src):
            shutil.copy(src, dst)
        return True
    images_rdd.map(copy_image).collect()

    spark.stop()

if __name__ == "__main__":
    main()
    
# spark-submit dataset_extractor.py \
#     --categories 12 24 57 \
#     --output_dir /path/to/output \
#     --original_dir /path/to/object365 \
#     --min_count 1000
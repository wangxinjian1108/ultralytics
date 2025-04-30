import os
import shutil
import json
import argparse
from pyspark.sql import SparkSession
from glob import glob

# 初始化 Spark
spark = SparkSession.builder.appName("Object365SubsetExtractor").getOrCreate()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract a subset from the Object365 dataset")
    parser.add_argument('--object_index', type=int, nargs='+', required=True, help="List of object indices to filter")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save extracted data")
    parser.add_argument('--input_dir', type=str, required=True, help="Original Object365 dataset directory")
    return parser.parse_args()

def filter_and_copy_files(input_dir, output_dir, object_index):
    # 获取所有图片路径和标签路径
    image_paths = glob(os.path.join(input_dir, "images", "*.jpg"))
    label_paths = glob(os.path.join(input_dir, "labels", "*.json"))
    
    # 准备一个集合，包含所有需要的图像和标签索引
    selected_images = set()
    selected_labels = set()

    for label_path in label_paths:
        with open(label_path, 'r') as f:
            labels = json.load(f)
            for label in labels:
                if label['object_index'] in object_index:
                    # 获取图像路径
                    image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
                    selected_images.add(image_path)
                    selected_labels.add(label_path)

    # 使用 Spark 进行并行化操作，筛选并复制图像和标签
    rdd = spark.sparkContext.parallelize(list(selected_images))
    rdd.foreach(lambda image_path: shutil.copy(image_path, os.path.join(output_dir, os.path.basename(image_path))))

    rdd_labels = spark.sparkContext.parallelize(list(selected_labels))
    rdd_labels.foreach(lambda label_path: shutil.copy(label_path, os.path.join(output_dir, os.path.basename(label_path))))

def main():
    # 解析输入参数
    args = parse_arguments()

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 调用函数进行数据过滤和复制
    filter_and_copy_files(args.input_dir, args.output_dir, args.object_index)

if __name__ == "__main__":
    main()

import argparse
import os
import shutil
import random
from pyspark.sql import SparkSession
from typing import List, Set, Tuple, Dict

def main():
    parser = argparse.ArgumentParser(description="Extract YOLO subset from Object365")
    parser.add_argument('--categories', type=int, nargs='+', required=True, help='Target category IDs')
    parser.add_argument('--remapped_categories', type=int, nargs='+', required=True, help='Remapped category IDs')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--original_dir', type=str, required=True, help='Original dataset directory')
    parser.add_argument('--min_count', type=int, required=True, help='Minimum instances per category')
    args = parser.parse_args()

    spark = SparkSession.builder.appName("YOLOSubsetExtractor").getOrCreate()
    sc = spark.sparkContext

    # 准备目录路径
    original_images = os.path.join(args.original_dir, 'images/train')
    original_labels = os.path.join(args.original_dir, 'labels/train')
    
    # 验证categories和remapped_categories长度相同
    if len(args.categories) != len(args.remapped_categories):
        raise ValueError("The number of categories must match the number of remapped categories")
    
    # 创建类别映射字典
    category_mapping = dict(zip(args.categories, args.remapped_categories))
    target_categories = set(args.categories)
    
    print(f"Category mapping: {category_mapping}")

    # 获取所有标签文件路径
    label_files = [os.path.join(original_labels, f) for f in os.listdir(original_labels) if f.endswith('.txt')]
    print(f'Found in total {len(label_files)} labeling files')
    
    # take first 10000 imgs for experiment
    label_files = label_files[:10000]
    
    label_rdd = sc.parallelize(label_files, numSlices=1000)

    # 处理每个标签文件，返回（类别ID, (图像路径, 标签行内容, 映射后类别)）
    def process_label_file(label_path: str) -> List[Tuple[int, Tuple[str, str, int]]]:
        try:
            image_file = os.path.basename(label_path).replace('.txt', '.jpg')
            image_path = os.path.join(original_images, image_file)
            results = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    if class_id in target_categories:
                        # 添加映射后的类别ID
                        remapped_id = category_mapping[class_id]
                        results.append((class_id, (image_path, line, remapped_id)))
            return results
        except Exception as e:
            print(f"Error processing {label_path}: {str(e)}")
            return []

    # 得到所有目标类别的实例（类别ID, (图像路径, 行内容, 映射后类别)）
    instances_rdd = label_rdd.flatMap(process_label_file)

    # 按原始类别分组并采样
    grouped_rdd = instances_rdd.groupByKey()

    # 采样函数
    def sample_instances(group):
        class_id, instances = group
        instances = list(instances)
        sample_size = min(args.min_count, len(instances))
        return random.sample(instances, sample_size) if instances else []

    sampled_rdd = grouped_rdd.flatMap(sample_instances)

    # 收集所有需要的图像路径和对应的实例信息
    image_instances_rdd = sampled_rdd.map(lambda x: (x[0], (x[1], x[2])))  # (image_path, (original_line, remapped_id))
    
    # 按图像路径分组，合并同一图像的所有实例
    image_grouped_rdd = image_instances_rdd.groupByKey()
    
    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels/train'), exist_ok=True)
    
    # 日志目录，记录处理的文件信息
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    log_file = os.path.join(args.output_dir, 'logs', 'processed_files.txt')

    # 处理每个图像及其标签
    def process_image_with_instances(item):
        image_path, instances_data = item
        try:
            # 复制图像
            image_name = os.path.basename(image_path)
            dest_image = os.path.join(args.output_dir, 'images/train', image_name)
            if not os.path.exists(dest_image):
                shutil.copy(image_path, dest_image)

            # 处理对应标签
            label_name = image_name.replace('.jpg', '.txt')
            dest_label = os.path.join(args.output_dir, 'labels/train', label_name)

            # 重新生成标签内容，应用类别映射
            remapped_lines = []
            for line_data in instances_data:
                original_line, remapped_id = line_data
                parts = original_line.strip().split()
                # 替换类别ID为映射后的ID
                parts[0] = str(remapped_id)
                remapped_lines.append(' '.join(parts) + '\n')

            # 写入新的标签文件
            with open(dest_label, 'w') as f_out:
                f_out.writelines(remapped_lines)
                
            return (image_name, len(remapped_lines))
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return (image_name, f"ERROR: {str(e)}")

    # 并行执行处理操作
    processing_results = image_grouped_rdd.map(process_image_with_instances).collect()
    
    # 记录处理结果
    with open(log_file, 'w') as log:
        log.write(f"Processed {len(processing_results)} images\n")
        log.write(f"Category mapping: {category_mapping}\n\n")
        for result in processing_results:
            log.write(f"{result[0]}: {result[1]} instances\n")
    
    # 创建dataset.yaml文件
    yaml_content = f"""# YOLO dataset configuration
path: {args.output_dir}  # dataset root dir
train: images/train  # train images
val: images/train  # using same images for validation (you may want to create a separate val set)

# Classes
nc: {len(args.remapped_categories)}  # number of classes
names:  # class names
"""

    # 添加类名（这里用默认名称，实际应用中应替换为真实类名）
    for i in range(len(args.remapped_categories)):
        yaml_content += f"  {i}: Class_{i}\n"
    
    # 写入YAML文件
    with open(os.path.join(args.output_dir, 'dataset.yaml'), 'w') as yaml_file:
        yaml_file.write(yaml_content)
    
    print(f"Processing complete. Results logged to {log_file}")
    print(f"Dataset configuration saved to {os.path.join(args.output_dir, 'dataset.yaml')}")
    
    spark.stop()

if __name__ == "__main__":
    main()
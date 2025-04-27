from ultralytics import SAM
import argparse
import cv2
import os
import sys
import glob
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

def process_image(model, image_path, output_dir, prompts=None):
    """
    对单张图像应用SAM模型进行分割
    
    Args:
        model: SAM模型实例
        image_path: 输入图像路径
        output_dir: 输出目录路径
        prompts: 分割提示点或框（可选）
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 确保输入图像存在
        if not os.path.isfile(image_path):
            print(f"错误: 图像文件不存在: {image_path}")
            return False
            
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图像: {image_path}")
            return False
        
        # 提取文件名
        filename = os.path.basename(image_path)
        
        # 使用SAM模型处理图像
        if prompts:
            # 如果有prompts，传入相应参数
            if "points" in prompts:
                points, labels = prompts["points"]
                results = model(image_path, points=points, labels=labels)
            elif "boxes" in prompts:
                results = model(image_path, bboxes=prompts["boxes"])
            else:
                results = model(image_path)
        else:
            # 默认不传入prompts
            results = model(image_path)
        
        # 保存结果
        for i, result in enumerate(results):
            # 获取分割结果
            if hasattr(result, 'plot'):
                segmented_img = result.plot()
                
                # 保存分割结果
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, segmented_img)
        
        return True
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_video(image_dir, output_video_path, fps=30):
    """
    从处理后的图像创建视频
    
    Args:
        image_dir: 包含处理后图像的目录
        output_video_path: 输出视频路径
        fps: 视频帧率
    
    Returns:
        bool: 是否成功创建视频
    """
    try:
        # 获取图像文件列表
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print("目录中没有找到图像")
            return False
        
        # 创建临时目录用于顺序帧
        temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 复制并重命名帧
        print("准备视频帧...")
        for i, img_file in enumerate(tqdm(image_files, desc="准备帧")):
            img_path = os.path.join(image_dir, img_file)
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imwrite(frame_path, img)
        
        # 确保输出路径有.mp4后缀
        video_path = os.path.splitext(output_video_path)[0] + '.mp4'
        
        # 使用ffmpeg创建视频
        print("使用ffmpeg创建视频...")
        ffmpeg_cmd = f"ffmpeg -y -framerate {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}"
        os.system(ffmpeg_cmd)
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print(f"成功创建视频: {video_path}")
        return True
        
    except Exception as e:
        print(f"创建视频时出错: {str(e)}")
        # 如果临时目录存在则清理
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def process_directory(model_path, input_dir, output_dir, fps=30, generate_video=True, prompts=None):
    """
    处理目录中的所有图像并可选择创建视频
    
    Args:
        model_path: SAM模型路径
        input_dir: 输入图像目录
        output_dir: 输出目录
        fps: 视频帧率
        generate_video: 是否生成视频
        prompts: 分割提示点或框（可选）
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载SAM模型
        print(f"加载SAM模型: {model_path}")
        model = SAM(model_path)
        
        # 获取图像文件列表
        image_files = sorted([f for f in os.listdir(input_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"在 {input_dir} 中没有找到图像")
            return False
            
        # 处理每个图像
        print("处理图像...")
        failed_images = []
        for image_file in tqdm(image_files, desc="处理图像"):
            image_path = os.path.join(input_dir, image_file)
            success = process_image(model, image_path, output_dir, prompts)
            if not success:
                failed_images.append(image_file)
        
        # 报告处理结果
        total_images = len(image_files)
        processed_images = total_images - len(failed_images)
        print(f"处理完成: {processed_images}/{total_images} 图像成功处理")
        if failed_images:
            print(f"无法处理的图像 ({len(failed_images)}):")
            for img in failed_images[:min(10, len(failed_images))]:
                print(f"  - {img}")
            if len(failed_images) > 10:
                print(f"  ... 以及另外 {len(failed_images) - 10} 个文件")
        
        # 如果需要，创建视频
        if generate_video and processed_images > 0:
            print("创建视频...")
            video_path = os.path.join(output_dir, "sam_video")
            if create_video(output_dir, video_path, fps):
                print(f"视频创建完成")
            else:
                print("创建视频失败")
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='用SAM模型处理目录中的图像并创建视频')
    parser.add_argument('--model', type=str, required=True, help='SAM模型路径')
    parser.add_argument('--input-dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--fps', type=int, default=30, help='输出视频的帧率 (默认: 30)')
    parser.add_argument('--no-video', action='store_true', help='不生成视频')
    parser.add_argument('--prompt-points', type=str, help='分割提示点，格式为x1,y1,x2,y2,... (可选)')
    
    args = parser.parse_args()
    
    # 解析提示点
    prompts = None
    if args.prompt_points:
        try:
            points = [float(p) for p in args.prompt_points.split(',')]
            if len(points) % 2 == 0:  # 确保坐标成对
                # 将坐标转换为形状 [N, 2] 的数组
                points = np.array(points).reshape(-1, 2)
                labels = np.ones(len(points))  # 默认所有点都是前景
                prompts = {"points": [points, labels]}
                print(f"使用提示点: {points}")
            else:
                print("错误: 提示点必须是成对的x,y坐标")
                sys.exit(1)
        except Exception as e:
            print(f"解析提示点时出错: {str(e)}")
            sys.exit(1)
    
    success = process_directory(
        args.model, 
        args.input_dir, 
        args.output_dir, 
        args.fps, 
        not args.no_video,
        prompts
    )
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
花卉分类模型预测脚本

使用方法:
    python ./code/predict.py <测试集文件夹> <输出文件路径>

示例:
    python ./code/predict.py ./unified_flower_dataset/images/test ./results/submission.csv

输出格式:
    CSV文件包含三列: filename, category_id, confidence
    - filename: 测试图片文件名
    - category_id: 预测的类别ID (对应花卉类别编号)
    - confidence: 预测置信度 (0-1之间)
"""

import os
import csv
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import logging

# 配置日志记录器
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def load_category_mapping(csv_path):
    """加载类别映射：英文名称 -> category_id"""
    name_to_id = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_to_id[row['english_name']] = int(row['category_id'])
    return name_to_id


def get_image_files(img_dir, img_extensions=None):
    """获取目录中的所有图片文件"""
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    img_dir_path = Path(img_dir)
    image_files = []

    for ext in img_extensions:
        image_files.extend(img_dir_path.glob(f'*{ext}'))
        image_files.extend(img_dir_path.glob(f'*{ext.upper()}'))

    # 只保留文件名，并排序
    image_files = sorted([f.name for f in image_files])
    return image_files


def predict_images(model, img_dir, image_files, name_to_id):
    """批量预测图片"""
    predictions = []

    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(img_dir, filename)

        try:
            results = model(image_path)

            for result in results:
                probs = result.probs
                confidence = float(probs.top1conf)
                english_name = result.names[probs.top1]
                category_id = name_to_id.get(english_name)

                if category_id is not None:
                    predictions.append({
                        'filename': filename,
                        'category_id': category_id,
                        'confidence': confidence
                    })
                else:
                    logger.error(f"警告: '{english_name}' 未在映射表中找到 (文件: {filename})")

        except Exception as e:
            logger.error(f"错误: 预测 {filename} 失败 - {e}")

        # 显示进度
        if i % 100 == 0:
            logger.error(f"已处理 {i}/{len(image_files)} 张图片...")

    return predictions


def main():
    parser = argparse.ArgumentParser(description='花卉分类模型预测')

    parser.add_argument('test_img_dir', type=str,
                        help='测试图片目录')
    parser.add_argument('output_path', type=str,
                        help='预测结果输出路径 (CSV文件)')

    args = parser.parse_args()

    logger.error(f'测试集目录: {args.test_img_dir}')
    logger.error(f'输出文件: {args.output_path}')

    logger.error("")

    # 检查路径
    if not os.path.exists(args.test_img_dir):
        logger.error(f"错误: 测试集目录不存在: {args.test_img_dir}")
        return

    # 加载类别映射
    logger.error("加载类别映射...")
    name_to_id = load_category_mapping('../src/category_mapping.csv')
    logger.error(f"已加载 {len(name_to_id)} 个类别映射")
    logger.error("")

    # 加载模型
    logger.error("加载模型...")
    model = YOLO('../model/runs/classify/train/weights/best.pt')
    logger.error("模型加载完成")
    logger.error("")

    # 获取图片文件
    logger.error("扫描测试集目录...")
    image_files = get_image_files(args.test_img_dir)

    if not image_files:
        logger.error(f"错误: 在目录 {args.test_img_dir} 中未找到图片文件")
        return

    logger.error(f"找到 {len(image_files)} 张图片")
    logger.error("")

    # 批量预测
    logger.error("开始预测...")
    predictions = predict_images(model, args.test_img_dir, image_files, name_to_id)
    logger.error(f"预测完成，成功预测 {len(predictions)} 张图片")
    logger.error("")

    # 保存结果
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('filename').reset_index(drop=True)

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.error(f"创建输出目录: {output_dir}")

    results_df.to_csv(args.output_path, index=False)
    logger.error(f"预测结果已保存到: {args.output_path}")
    logger.error("")

    # 显示统计信息
    logger.error("预测统计:")
    logger.error(f"  总图片数: {len(image_files)}")
    logger.error(f"  成功预测: {len(predictions)}")
    logger.error(f"  平均置信度: {results_df['confidence'].mean():.4f}")
    logger.error("")
    logger.error("预测完成!")


if __name__ == '__main__':
    main()
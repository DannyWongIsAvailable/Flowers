#!/usr/bin/env python3
"""
花卉分类数据集工具函数

包含功能:
1. 提取类别映射 (extract_category_mapping)
2. 划分训练集和验证集 (organize_and_split_dataset)
"""

import shutil
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict


def extract_category_mapping(labels_file='../src/train_labels.csv',
                             output_file='../src/category_mapping.csv'):
    """
    从train_labels.csv提取唯一的category_id和english_name映射

    参数:
        labels_file: 标签文件路径
        output_file: 输出映射文件路径
    """
    labels_file = Path(labels_file)
    output_file = Path(output_file)

    if not labels_file.exists():
        print(f"错误: 标签文件 {labels_file} 不存在!")
        return

    print("=" * 70)
    print("提取类别ID映射")
    print("=" * 70)

    # 读取CSV文件
    print(f"正在读取: {labels_file}")
    df = pd.read_csv(labels_file)
    df.columns = df.columns.str.strip()

    print(f"总记录数: {len(df)}")

    # 提取唯一的category_id和english_name组合
    mapping_df = df[['category_id', 'english_name']].drop_duplicates()
    mapping_df = mapping_df.sort_values('category_id').reset_index(drop=True)
    mapping_df['english_name'] = mapping_df['english_name'].str.strip()

    print(f"唯一类别数: {len(mapping_df)}")

    # 检查重复的category_id
    duplicates = mapping_df.groupby('category_id')['english_name'].nunique()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        print(f"\n⚠️  警告: 发现 {len(duplicates)} 个category_id对应多个english_name")
        print("   将保留第一个出现的english_name")
        mapping_df = mapping_df.drop_duplicates(subset='category_id', keep='first')

    # 保存到CSV
    mapping_df.to_csv(output_file, index=False)
    print(f"\n✅ 保存成功: {output_file.absolute()}")
    print(f"   最小category_id: {mapping_df['category_id'].min()}")
    print(f"   最大category_id: {mapping_df['category_id'].max()}")
    print(f"   总类别数: {len(mapping_df)}")
    print("=" * 70)


def organize_and_split_dataset(src_dir='../src/train',
                               labels_file='../src/train_labels.csv',
                               train_dir='../dataset/flowers_cls/train',
                               val_dir='../dataset/flowers_cls/val',
                               train_ratio=0.9,
                               seed=42):
    """
    优化版本：先分配数据再复制文件

    参数:
        src_dir: 源图片目录
        labels_file: 标签CSV文件路径
        train_dir: 训练集输出目录
        val_dir: 验证集输出目录
        train_ratio: 训练集比例，默认0.9
        seed: 随机种子，确保可复现
    """
    random.seed(seed)

    src_dir = Path(src_dir)
    labels_file = Path(labels_file)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)

    # 检查路径
    if not src_dir.exists():
        print(f"错误: 源文件夹 {src_dir} 不存在!")
        return

    if not labels_file.exists():
        print(f"错误: 标签文件 {labels_file} 不存在!")
        return

    print("=" * 70)
    print("步骤 1/3: 读取标签文件并分组")
    print("=" * 70)

    # 读取CSV文件
    df = pd.read_csv(labels_file)
    df.columns = df.columns.str.strip()

    # 按english_name分组
    grouped = defaultdict(list)
    missing_files = []

    print(f"总共 {len(df)} 个文件记录")

    for idx, row in df.iterrows():
        filename = row['filename'].strip()
        english_name = row['english_name'].strip()
        safe_english_name = english_name.replace('/', '_').replace('\\', '_').replace(':', '_')

        src_file = src_dir / filename
        if src_file.exists():
            grouped[safe_english_name].append(src_file)
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"⚠️  警告: 有 {len(missing_files)} 个文件不存在")

    print(f"找到 {len(grouped)} 个类别")
    print(f"有效文件: {sum(len(files) for files in grouped.values())} 个")

    print("\n" + "=" * 70)
    print(f"步骤 2/3: 划分数据集 (训练:{train_ratio * 100:.0f}% / 验证:{(1 - train_ratio) * 100:.0f}%)")
    print("=" * 70)

    # 在内存中完成划分
    train_split = defaultdict(list)
    val_split = defaultdict(list)
    stats = defaultdict(lambda: {'train': 0, 'val': 0})

    for category_name, image_files in sorted(grouped.items()):
        image_files_copy = image_files.copy()
        random.shuffle(image_files_copy)

        n_total = len(image_files_copy)
        n_train = int(n_total * train_ratio)

        # 确保验证集至少有1个样本
        if n_total > 1 and n_train == n_total:
            n_train = n_total - 1

        train_split[category_name] = image_files_copy[:n_train]
        val_split[category_name] = image_files_copy[n_train:]

        stats[category_name]['train'] = len(train_split[category_name])
        stats[category_name]['val'] = len(val_split[category_name])

    total_train = sum(stats[cat]['train'] for cat in stats)
    total_val = sum(stats[cat]['val'] for cat in stats)

    print(f"划分完成:")
    print(f"  训练集: {total_train} 张 ({total_train / (total_train + total_val) * 100:.1f}%)")
    print(f"  验证集: {total_val} 张 ({total_val / (total_train + total_val) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("步骤 3/3: 复制文件到目标目录")
    print("=" * 70)

    # 创建目录
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    copied_train = 0
    copied_val = 0
    total_files = total_train + total_val

    # 复制训练集
    print("正在复制训练集文件...")
    for category_name, image_files in train_split.items():
        train_category_dir = train_dir / category_name
        train_category_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            target_file = train_category_dir / img_file.name
            shutil.copy2(img_file, target_file)
            copied_train += 1

            if copied_train % 500 == 0:
                print(f"  进度: {copied_train}/{total_files} ({copied_train / total_files * 100:.1f}%)")

    print(f"  ✅ 训练集复制完成: {copied_train} 个文件")

    # 复制验证集
    print("\n正在复制验证集文件...")
    for category_name, image_files in val_split.items():
        val_category_dir = val_dir / category_name
        val_category_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            target_file = val_category_dir / img_file.name
            shutil.copy2(img_file, target_file)
            copied_val += 1

            if copied_val % 500 == 0:
                progress = (copied_train + copied_val) / total_files * 100
                print(f"  进度: {copied_train + copied_val}/{total_files} ({progress:.1f}%)")

    print(f"  ✅ 验证集复制完成: {copied_val} 个文件")

    # 输出最终统计
    print("\n" + "=" * 70)
    print("✅ 全部完成!")
    print("=" * 70)
    print(f"训练集: {copied_train} 张 ({copied_train / (copied_train + copied_val) * 100:.1f}%)")
    print(f"验证集: {copied_val} 张 ({copied_val / (copied_train + copied_val) * 100:.1f}%)")
    print(f"总计: {copied_train + copied_val} 张")
    print(f"类别数: {len(stats)}")
    print(f"\n训练集目录: {train_dir.absolute()}")
    print(f"验证集目录: {val_dir.absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    # 使用示例

    # 1. 提取类别映射
    print("任务 1: 提取类别映射\n")
    extract_category_mapping()

    print("\n\n")

    # 2. 划分数据集
    print("任务 2: 划分数据集\n")
    organize_and_split_dataset(
        train_ratio=0.9,
        seed=42
    )
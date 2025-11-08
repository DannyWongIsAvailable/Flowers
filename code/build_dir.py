import shutil
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict


def organize_and_split_dataset(train_ratio=0.9, seed=42):
    """
    优化版本：先分配数据再复制文件
    1. 读取CSV并按类别分组
    2. 在内存中完成train/val划分
    3. 一次性复制文件到对应目录

    参数:
        train_ratio: 训练集比例，默认0.9
        seed: 随机种子，确保可复现
    """
    # 设置随机种子
    random.seed(seed)

    # 定义路径
    src_dir = Path('../src/train')
    labels_file = Path('../src/train_labels.csv')
    train_dir = Path('../dataset/flowers_cls/train')
    val_dir = Path('../dataset/flowers_cls/val')

    # 检查源文件夹和标签文件是否存在
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
    print("正在读取标签文件...")
    df = pd.read_csv(labels_file)

    # 清理列名中的空格
    df.columns = df.columns.str.strip()

    # 按english_name分组
    grouped = defaultdict(list)
    missing_files = []

    print(f"总共 {len(df)} 个文件记录")
    print("正在检查文件并分组...\n")

    for idx, row in df.iterrows():
        filename = row['filename'].strip()
        english_name = row['english_name'].strip()

        # 替换英文名称中的不合法字符
        safe_english_name = english_name.replace('/', '_').replace('\\', '_').replace(':', '_')

        # 源文件路径
        src_file = src_dir / filename

        if src_file.exists():
            grouped[safe_english_name].append(src_file)
        else:
            missing_files.append(filename)

    if missing_files:
        print(f"⚠️  警告: 有 {len(missing_files)} 个文件不存在")
        if len(missing_files) <= 10:
            for f in missing_files:
                print(f"   - {f}")

    print(f"\n找到 {len(grouped)} 个类别")
    print(f"有效文件: {sum(len(files) for files in grouped.values())} 个")

    print("\n" + "=" * 70)
    print(f"步骤 2/3: 划分数据集 (训练集:{train_ratio * 100:.0f}% / 验证集:{(1 - train_ratio) * 100:.0f}%)")
    print("=" * 70)
    print(f"随机种子: {seed}\n")

    # 在内存中完成划分
    train_split = defaultdict(list)
    val_split = defaultdict(list)
    stats = defaultdict(lambda: {'train': 0, 'val': 0})

    for category_name, image_files in sorted(grouped.items()):
        # 打乱文件顺序
        image_files_copy = image_files.copy()
        random.shuffle(image_files_copy)

        # 计算划分点
        n_total = len(image_files_copy)
        n_train = int(n_total * train_ratio)

        # 确保验证集至少有1个样本（如果总数大于1）
        if n_total > 1 and n_train == n_total:
            n_train = n_total - 1

        # 划分文件
        train_split[category_name] = image_files_copy[:n_train]
        val_split[category_name] = image_files_copy[n_train:]

        # 更新统计
        stats[category_name]['train'] = len(train_split[category_name])
        stats[category_name]['val'] = len(val_split[category_name])

    # 计算总数
    total_train = sum(stats[cat]['train'] for cat in stats)
    total_val = sum(stats[cat]['val'] for cat in stats)

    print(f"划分完成:")
    print(f"  训练集: {total_train} 张图片 ({total_train / (total_train + total_val) * 100:.1f}%)")
    print(f"  验证集: {total_val} 张图片 ({total_val / (total_train + total_val) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("步骤 3/3: 复制文件到目标目录")
    print("=" * 70)

    # 创建train和val目录
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    copied_train = 0
    copied_val = 0
    total_files = total_train + total_val

    # 复制训练集文件
    print("正在复制训练集文件...")
    for category_name, image_files in train_split.items():
        # 创建目标类别文件夹
        train_category_dir = train_dir / category_name
        train_category_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            target_file = train_category_dir / img_file.name
            shutil.copy2(img_file, target_file)
            copied_train += 1

            # 显示进度
            if copied_train % 100 == 0:
                progress = (copied_train / total_files) * 100
                print(f"  进度: {copied_train}/{total_files} ({progress:.1f}%)")

    print(f"  训练集复制完成: {copied_train} 个文件")

    # 复制验证集文件
    print("\n正在复制验证集文件...")
    for category_name, image_files in val_split.items():
        # 创建目标类别文件夹
        val_category_dir = val_dir / category_name
        val_category_dir.mkdir(parents=True, exist_ok=True)

        for img_file in image_files:
            target_file = val_category_dir / img_file.name
            shutil.copy2(img_file, target_file)
            copied_val += 1

            # 显示进度
            if copied_val % 100 == 0:
                progress = ((copied_train + copied_val) / total_files) * 100
                print(f"  进度: {copied_train + copied_val}/{total_files} ({progress:.1f}%)")

    print(f"  验证集复制完成: {copied_val} 个文件")

    # 输出最终统计
    print("\n" + "=" * 70)
    print("全部完成!")
    print("=" * 70)
    print(f"训练集: {copied_train} 张图片 ({copied_train / (copied_train + copied_val) * 100:.1f}%)")
    print(f"验证集: {copied_val} 张图片 ({copied_val / (copied_train + copied_val) * 100:.1f}%)")
    print(f"总计: {copied_train + copied_val} 张图片")
    print(f"类别数量: {len(stats)}")
    print(f"\n训练集目录: {train_dir.absolute()}")
    print(f"验证集目录: {val_dir.absolute()}")
    print("=" * 70)

    # 显示每个类别的详细分布（类别较少时显示）
    if len(stats) <= 20:
        print("\n各类别分布:")
        for cat_name in sorted(stats.keys()):
            counts = stats[cat_name]
            total = counts['train'] + counts['val']
            print(f"  {cat_name}: {counts['train']} 训练 / {counts['val']} 验证 (总计: {total})")

    # 检查异常情况
    warnings = []
    for cat_name, counts in stats.items():
        if counts['val'] == 0:
            warnings.append(f"⚠️  {cat_name}: 没有验证集样本")
        if counts['train'] == 0:
            warnings.append(f"⚠️  {cat_name}: 没有训练集样本")

    if warnings:
        print("\n异常检查:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("\n✅ 所有类别都有训练集和验证集样本")


if __name__ == '__main__':
    # 可以修改这些参数
    organize_and_split_dataset(
        train_ratio=0.9,  # 训练集比例
        seed=42  # 随机种子
    )
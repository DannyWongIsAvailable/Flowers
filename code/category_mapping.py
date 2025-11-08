import pandas as pd
from pathlib import Path


def extract_category_mapping():
    """
    从train_labels.csv提取唯一的category_id和english_name映射
    输出为category_mapping.csv
    """
    # 定义路径
    labels_file = Path('../src/train_labels.csv')
    output_file = Path('../src/category_mapping.csv')

    # 检查源文件是否存在
    if not labels_file.exists():
        print(f"错误: 标签文件 {labels_file} 不存在!")
        return

    print("=" * 70)
    print("提取类别ID映射")
    print("=" * 70)

    # 读取CSV文件
    print(f"正在读取: {labels_file}")
    df = pd.read_csv(labels_file)

    # 清理列名中的空格
    df.columns = df.columns.str.strip()

    print(f"总记录数: {len(df)}")

    # 提取唯一的category_id和english_name组合
    # 使用drop_duplicates确保唯一性
    mapping_df = df[['category_id', 'english_name']].drop_duplicates()

    # 按category_id排序
    mapping_df = mapping_df.sort_values('category_id').reset_index(drop=True)

    # 清理english_name中的空格
    mapping_df['english_name'] = mapping_df['english_name'].str.strip()

    print(f"唯一类别数: {len(mapping_df)}")

    # 检查是否有category_id对应多个english_name的情况
    duplicates = mapping_df.groupby('category_id')['english_name'].nunique()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        print(f"\n⚠️  警告: 发现 {len(duplicates)} 个category_id对应多个english_name:")
        for cat_id in duplicates.index:
            names = mapping_df[mapping_df['category_id'] == cat_id]['english_name'].tolist()
            print(f"   category_id {cat_id}: {names}")
        print("   将保留第一个出现的english_name")
        # 保留每个category_id的第一个记录
        mapping_df = mapping_df.drop_duplicates(subset='category_id', keep='first')

    # 保存到CSV
    mapping_df.to_csv(output_file, index=False)

    print(f"\n保存成功: {output_file.absolute()}")
    print("=" * 70)

    # 显示前10条记录作为预览
    print("\n预览前10条记录:")
    print(mapping_df.head(10).to_string(index=False))

    if len(mapping_df) > 10:
        print(f"\n... (共 {len(mapping_df)} 条记录)")

    # 显示统计信息
    print(f"\n统计信息:")
    print(f"  最小category_id: {mapping_df['category_id'].min()}")
    print(f"  最大category_id: {mapping_df['category_id'].max()}")
    print(f"  总类别数: {len(mapping_df)}")


if __name__ == '__main__':
    extract_category_mapping()
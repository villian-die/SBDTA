import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file, train_file, val_file, test_file, test_size=0.15, val_size=0.15, random_state=42):
    """
    将数据集划分为训练集、验证集和测试集。

    参数:
    - input_file: 原始数据文件路径
    - train_file: 训练集文件保存路径
    - val_file: 验证集文件保存路径
    - test_file: 测试集文件保存路径
    - test_size: 测试集占总数据集的比例
    - val_size: 验证集占训练集的比例
    - random_state: 随机种子
    """
    # 读取原始数据
    df = pd.read_csv(input_file)

    # 划分训练集和测试集
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # 从训练集中划分验证集
    train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), random_state=random_state)

    # 保存划分后的数据集
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"训练集已保存到 {train_file}")
    print(f"验证集已保存到 {val_file}")
    print(f"测试集已保存到 {test_file}")

if __name__ == "__main__":
    # 设置文件路径
    input_file = "datasets/kiba/raw/data.csv"  # 原始数据文件路径
    train_file = "datasets/kiba/raw/data_train.csv"  # 训练集文件路径
    val_file = "datasets/kiba/raw/data_val.csv"  # 验证集文件路径
    test_file = "datasets/kiba/raw/data_test.csv"  # 测试集文件路径

    # 调用函数进行数据划分
    split_data(input_file, train_file, val_file, test_file)

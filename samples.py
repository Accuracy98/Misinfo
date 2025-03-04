import json

# 加载JSON文件
file_path = "merged_data.json"

try:
    with open(file_path, "r", encoding="utf-8") as file:
        train_dataset = json.load(file)
    
    # 计算样本数量
    num_samples = len(train_dataset)
    print(f"数据集中包含 {num_samples} 个样本。")
except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到。请确保文件路径正确。")
except json.JSONDecodeError:
    print(f"错误：文件 {file_path} 不是一个有效的JSON文件。")
except Exception as e:
    print(f"发生错误：{e}")
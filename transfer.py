import json
import csv

# 假设JSON数据存储在json文件中，名为"data.json"
input_file = "test_loss.json"
output_file = "test_loss.csv"

# 打开并读取JSON文件
with open(input_file, "r") as f:
    data = [json.loads(line) for line in f]

# 提取"loss"（包含train_loss处理）和"epoch"字段
filtered_data = []
for entry in data:
    # 处理train_loss字段，统一命名为loss
    loss = entry.get("train_loss", entry.get("loss"))
    epoch = entry.get("epoch")
    if loss is not None and epoch is not None:
        filtered_data.append({"loss": loss, "epoch": epoch})

# 将提取的数据写入CSV文件
with open(output_file, "w", newline="") as csvfile:
    fieldnames = ["loss", "epoch"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入表头
    writer.writeheader()
    
    # 写入数据
    writer.writerows(filtered_data)

print(f"CSV file saved as {output_file}")

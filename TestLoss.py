import csv
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

# 定义路径
fine_tuned_model_path = "./llama-3.2-3b-instruct-finetuned"  # 训练好的模型路径
test_dataset_path = "./test_dataset.json"  # 测试集路径
test_loss_csv_path = "test_loss.csv"  # 输出 CSV 文件

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, trust_remote_code=True)

# 加载测试数据集
test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")

# 预处理：Tokenization，**添加 labels**
def tokenize_function(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()  # 这里添加 labels，保证 loss 计算
    return tokenized

test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 创建 DataLoader，确保 batch 是 tensor
def collate_fn(batch):
    return {key: torch.stack([torch.tensor(d[key]) for d in batch]) for key in batch[0]}

test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

# 计算测试集 loss
model.eval()
total_loss = 0
num_batches = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}  # 确保 batch 在正确的设备上
        outputs = model(**batch)

        # 确保 outputs.loss 存在
        if outputs.loss is not None:
            loss = outputs.loss.item()
            total_loss += loss
            num_batches += 1

# 计算平均测试集 loss
avg_test_loss = total_loss / num_batches if num_batches > 0 else float("nan")
print(f"Average Test Loss: {avg_test_loss}")

# 记录到 CSV
with open(test_loss_csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "test_loss"])
    writer.writerow([1, avg_test_loss])  # 这里只计算一个 epoch，如果有多个 epoch 可以循环记录

print(f"Test loss saved to {test_loss_csv_path}")

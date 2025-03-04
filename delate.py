import json
import re

# 你的数据集文件名
INPUT_FILE = "results.json"
OUTPUT_FILE = "new_result.json"

# 正则表达式匹配 "(Screenshot from ...)" 形式的文本
screenshot_pattern = re.compile(r'\(Screenshot from .*?\)')

def clean_claims(claims):
    """ 去除 claims 列表中的所有 '(Screenshot from ...)' 片段 """
    return [screenshot_pattern.sub('', claim).strip() for claim in claims]

def process_dataset(input_file, output_file):
    # 读取 JSON 文件
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历所有数据项，清理 `claims` 字段
    for item in data:
        if "claims" in item and isinstance(item["claims"], list):
            item["claims"] = clean_claims(item["claims"])

    # 保存清理后的 JSON 数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成！清理后的数据已保存到 {output_file}")

# 运行数据处理
if __name__ == "__main__":
    process_dataset(INPUT_FILE, OUTPUT_FILE)


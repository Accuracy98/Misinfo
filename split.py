import json
from sklearn.model_selection import train_test_split

# Load the dataset
with open('modified.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ensure the data format is correct
if isinstance(data, list) and all('text' in sample for sample in data):
    print(f"Dataset loaded successfully, with {len(data)} samples.")
else:
    raise ValueError("Incorrect data format, each sample should be a dictionary containing 'text'.")

# Split the dataset (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the split datasets
with open('train_dataset.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

with open('test_dataset.json', 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, ensure_ascii=False, indent=4)

print(f"Training set saved to train_dataset.json, with {len(train_data)} samples.")
print(f"Testing set saved to test_dataset.json, with {len(test_data)} samples.")

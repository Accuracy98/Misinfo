import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm  # Import tqdm for the progress bar

# Define model paths
models = {
    "DeepSeek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Base Llama": "meta-llama/Llama-3.1-8B-Instruct",
    "Fine-Tuned Llama": "./llama-3.1-8b-instruct-finetuned"
}

tokenizers = {}
loaded_models = {}

# Load models and tokenizers
for model_name, model_path in models.items():
    print(f"Loading {model_name} model...")
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")

# Load the test dataset
test_dataset_path = "./test_dataset.json"
test_data = load_dataset("json", data_files=test_dataset_path, split="train")


def generate_responses(model, tokenizer, test_data, max_input_length=128, max_new_tokens=50):
    responses = []
    device = next(model.parameters()).device  # Get the device where the model is located
    for sample in tqdm(test_data, desc="Generating responses", unit="sample"):  # Progress bar for generating responses
        input_text = sample["text"]  # Input text from the test dataset
        
        # Tokenize input text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length  # Limit input length
        )
        
        # Move inputs to the model's device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(generated_text)
    return responses


def compute_rouge_scores(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, hyp in tqdm(zip(references, hypotheses), desc="Computing ROUGE scores", unit="pair"):  # Progress bar for computing ROUGE scores
        rouge_score = scorer.score(ref, hyp)
        for metric in scores:
            scores[metric].append(rouge_score[metric].fmeasure)

    # Calculate average scores
    avg_scores = {metric: sum(values) / len(values) for metric, values in scores.items()}
    return avg_scores

# Get the reference answers from the test dataset
references = [sample["text"] for sample in test_data]

# Generate responses and compute ROUGE scores for each model
rouge_results = {}
for model_name, model in loaded_models.items():
    print(f"Generating responses with {model_name}...")
    responses = generate_responses(model, tokenizers[model_name], test_data, max_input_length=128, max_new_tokens=50)
    
    print(f"Computing ROUGE scores for {model_name}...")
    rouge_scores = compute_rouge_scores(references, responses)
    rouge_results[model_name] = rouge_scores
    print(f"{model_name} ROUGE scores: {rouge_scores}")

# Save the results to a text file
with open("DeepSeek_Llama_LlamaFinetuned_ROUGE.txt", "w") as output_file:
    for model_name, scores in rouge_results.items():
        output_file.write(f"{model_name} ROUGE scores:\n")
        output_file.write(str(scores) + "\n\n")

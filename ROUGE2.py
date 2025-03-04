import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the models and tokenizer
base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
fine_tuned_model_path = "./llama-3.1-8b-instruct-finetuned"
test_dataset_path = "./test_dataset.json"

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load the test dataset
test_data = load_dataset("json", data_files=test_dataset_path, split="train")


def generate_responses(model, tokenizer, test_data, max_input_length=128, max_new_tokens=50):
    responses = []
    device = next(model.parameters()).device  # Get the device where the model is located
    for sample in tqdm(test_data, desc="Generating responses", unit="sample"):  # Progress bar for generating responses
        input_text = sample["text"]  # Input text from the test dataset
        # Truncate input text to fit within the model's input limit
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length  # Limit input length
        )
        # Move inputs to the model's device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Use max_new_tokens instead of max_length
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

# Generate responses from the base model
print("Generating responses with base model...")
base_responses = generate_responses(base_model, tokenizer, test_data, max_input_length=128, max_new_tokens=50)

# Generate responses from the fine-tuned model
print("Generating responses with fine-tuned model...")
fine_tuned_responses = generate_responses(fine_tuned_model, tokenizer, test_data, max_input_length=128, max_new_tokens=50)

# Compute ROUGE scores
print("Computing ROUGE scores for base model...")
base_rouge_scores = compute_rouge_scores(references, base_responses)
print(f"Base model ROUGE scores: {base_rouge_scores}")

print("Computing ROUGE scores for fine-tuned model...")
fine_tuned_rouge_scores = compute_rouge_scores(references, fine_tuned_responses)
print(f"Fine-tuned model ROUGE scores: {fine_tuned_rouge_scores}")

# Save the results to a text file
with open("Llama_LlamaFinetuned_ROUGE.txt", "w") as output_file:
    output_file.write("Base model ROUGE scores:\n")
    output_file.write(str(base_rouge_scores) + "\n\n")
    
    output_file.write("Fine-tuned model ROUGE scores:\n")
    output_file.write(str(fine_tuned_rouge_scores) + "\n")

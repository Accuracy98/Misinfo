import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load the fine-tuned model and tokenizer
fine_tuned_model = "./llama-3.1-8b-instruct-finetuned"
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model,
    torch_dtype=torch.float16,  # Choose dtype based on your hardware
    device_map="auto"  # Automatically map to GPU or CPU
)

# Define the function to generate explanations
def generate_explanation(claim, max_length=360, temperature=0.7):
    # Construct the input prompt
    instruction = (
        "Provide a detailed explanation to help verify the following claim:\n\n"
        f"Claim: {claim}\n\n"
        "Explanation:"
    )
    
    # Convert the input prompt to model input
    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
    
    # Generate output using the model
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the generated text
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation

# Define the function for interactive interface
def interactive_interface(claim):
    explanation = generate_explanation(claim)
    return explanation

custom_css = """
h1 { font-size: 32px !important; }  /* Title font size */
h2 { font-size: 24px !important; }  /* Description font size */
input, textarea { font-size: 24px !important; }  /* Input & output text size */
"""

# Create a Gradio interface
interface = gr.Interface(
    fn=interactive_interface,
    inputs=gr.Textbox(label="Enter a Claim"),
    outputs=gr.Textbox(label="Generated Explanation"),
    title="Llama3.1-8B-Ins-Finetuned",
    description="Enter a claim, and the model will generate an explanation to help verify its credibility.",
    css=custom_css
)

# Launch the Gradio interface with a public sharing URL
interface.launch(share=True)

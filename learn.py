from transformers import T5ForConditionalGeneration, AutoTokenizer

# Load the model and tokenizer
model_name = "google/flan-t5-xl"
cache_dir = "./cache/models"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

def generate_response(prompt, max_length=512):
    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response
    outputs = model.generate(
        inputs, 
        max_length=max_length,
        num_beams=4,  # Beam search for better quality
        early_stopping=True,
        temperature=0.7
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "Translate the following English text to French: Hello, how are you?"
result = generate_response(prompt)
print(result)

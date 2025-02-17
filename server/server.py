from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"

# Initialize tokenizer and model
model = AutoModelForCausalLM.from_pretrained(
    "dicta-il/dictalm2.0-instruct", 
    torch_dtype=torch.bfloat16, 
    device_map=device,
    pad_token_id=2  # Explicitly set pad_token_id
)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")

# Configure tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'  # Ensure consistent padding

messages = [
    {"role": "user", "content": "איזה רוטב אהוב עליך?"},
    {"role": "assistant", "content": "טוב, אני די מחבב כמה טיפות מיץ לימון סחוט טרי. זה מוסיף בדיוק את הכמות הנכונה של טעם חמצמץ לכל מה שאני מבשל במטבח!"},
    {"role": "user", "content": "האם יש לך מתכונים למיונז?"}
]

# Create input with attention mask
encoded = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt",
    padding=True,
    add_special_tokens=True
).to(device)

# Create attention mask
attention_mask = torch.ones_like(encoded).to(device)

# Generate with attention mask
generated_ids = model.generate(
    encoded,
    attention_mask=attention_mask,
    max_new_tokens=4000,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True
)

decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and tokenizer
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(
    "dicta-il/dictalm2.0-instruct", 
    torch_dtype=torch.bfloat16, 
    device_map=device,
    pad_token_id=2
)
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictalm2.0-instruct")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Convert messages to list of dicts for the tokenizer
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Encode input
        encoded = tokenizer.apply_chat_template(
            messages_dict, 
            return_tensors="pt",
            padding=True,
            add_special_tokens=True
        ).to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(encoded).to(device)
        
        # Generate response
        generated_ids = model.generate(
            encoded,
            attention_mask=attention_mask,
            max_new_tokens=4000,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Decode response
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = decoded[0]
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
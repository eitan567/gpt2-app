from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging
import time
import re
from typing import List, Dict, Optional

load_dotenv()

# קונפיגורציה בסיסית
HF_TOKEN = os.getenv('HF_TOKEN')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ניהול שיחות
class ConversationManager:
    def __init__(self, max_history: int = 10, max_age_minutes: int = 60):
        self.conversations: Dict[str, List[Dict]] = {}
        self.max_history = max_history
        self.max_age_minutes = max_age_minutes
        
    def add_message(self, conversation_id: str, role: str, content: str):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            
        self.conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
            
    def get_conversation_context(self, conversation_id: str, max_tokens: int, tokenizer) -> str:
        if conversation_id not in self.conversations:
            return ""
            
        current_time = time.time()
        context_messages = []
        total_length = 0
        
        for msg in self.conversations[conversation_id]:
            message_age_minutes = (current_time - msg["timestamp"]) / 60
            if message_age_minutes > self.max_age_minutes:
                continue
                
            message_text = f"{msg['role']}: {msg['content']}\n"
            message_tokens = len(tokenizer.encode(message_text))
            
            if total_length + message_tokens > max_tokens:
                break
                
            context_messages.append(message_text)
            total_length += message_tokens
            
        return "".join(context_messages)
        
    def clear_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

# FastAPI setup
app = FastAPI()
conversation_manager = ConversationManager()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# מודל וטוקנייזר
MODEL_NAME = "dicta-il/dictalm2.0-instruct"
# הגדרת קונפיגורציית הקוונטיזציה
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# אתחול המודל עם הקונפיגורציה החדשה
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    quantization_config=quantization_config,
    token=HF_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Pydantic models
class TextGenerationRequest(BaseModel):
    input_text: str
    max_length: Optional[int] = Field(default=3072, ge=10, le=32768)  # שינוי ל-3072 כמו בקוד החדש
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)
    language: Optional[str] = Field(default="he", pattern="^(he|en)$")
    min_length: int = Field(default=10, ge=1, le=100)
    length_penalty: float = Field(default=1.0, ge=0.1, le=2.0)

def clean_response(text: str) -> str:
    # נקה את התגיות של ה-chat completion
    if "<|im_start|>assistant\n" in text:
        text = text.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
        
    # ניקוי תגיות ורווחים
    text = text.replace("<s>", "").replace("</s>", "")
    text = ' '.join(text.split())
    
    # הסרת משפטי פתיחה גנריים
    generic_starts = [
        "אני אשמח לעזור",
        "אני יכול לעזור",
        "אשמח לסייע",
        "אני כאן כדי לעזור",
        "אני מבין",
        "אני רואה",
        "בוודאי",
        "בהחלט"
    ]
    for start in generic_starts:
        if text.strip().lower().startswith(start.lower()):
            text = text.replace(start, "", 1)
            
    # ניקוי סימני פיסוק עודפים
    text = re.sub(r'([.!?])\1+', r'\1', text)
    
    # ניקוי רווחים מיותרים
    text = ' '.join(text.split())
    
    return text.strip()

def validate_response(response: str, question: str) -> bool:
    """בדיקה מקיפה של תקינות התשובה"""
    if not response or len(response.strip()) < 10:  # מינימום 10 תווים
        logger.warning("Response too short")
        return False
        
    # בדיקה שהתשובה מכילה מילים עבריות
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    if not hebrew_pattern.search(response):
        logger.warning("Response contains no Hebrew characters")
        return False
        
    # בדיקה שאין חזרות מיותרות
    words = response.split()
    if len(set(words)) < len(words) * 0.7:  # יותר מ-30% חזרות
        logger.warning("Response contains too many repetitions")
        return False
        
    # בדיקה שהתשובה לא מכילה שאריות מהפרומפט
    prompt_artifacts = [
        "אתה עוזר אישי",
        "תפקידך:",
        "בכל תשובה:",
        "system:",
        "assistant:",
        "user:"
    ]
    if any(artifact in response.lower() for artifact in prompt_artifacts):
        logger.warning("Response contains prompt artifacts")
        return False
        
    return True

def get_clean_conversation_context(conversation_id: str, max_tokens: int) -> str:
    if not conversation_id:
        return ""
        
    context = conversation_manager.get_conversation_context(
        conversation_id,
        max_tokens=max_tokens,
        tokenizer=tokenizer
    )
    
    if context:
        context = "היסטוריית שיחה קודמת:\n" + context
        
    return context

@app.post("/generate/")
async def generate_text(
    request: TextGenerationRequest,
    conversation_id: Optional[str] = Header(None, description="Unique conversation identifier")):

    try:
        max_model_length = model.config.max_position_embeddings
        logger.info(f"Model maximum context length: {max_model_length}")
        
        conversation_context = get_clean_conversation_context(
            conversation_id,
            max_tokens=max_model_length // 2
        )
        
        # הגדרת system prompt בפורמט המתאים
        system_prompt = """<|im_start|>system
                    אתה עוזר אישי מקצועי העונה בעברית. תפקידך לספק תשובות מדויקות ומועילות תוך שימוש בשפה ברורה ומכבדת.
                    <|im_end|>
                    משתמש: """

                       
        # בניית הפרומפט בפורמט OpenAI chat completion
        messages = []
        if conversation_context:
            previous_messages = conversation_context.strip().split('\n') if conversation_context else []

            for msg in previous_messages:
                if msg.startswith('user: '):
                    messages.append(f"<|im_start|>user\n{msg[6:]}<|im_end|>")
                elif msg.startswith('assistant: '):
                    messages.append(f"<|im_start|>assistant\n{msg[11:]}<|im_end|>")

            # הוספת ההודעה הנוכחית
            messages.append(f"<|im_start|>user\n{request.input_text.strip()}<|im_end|>")
            messages.append("<|im_start|>assistant\n")

            # חיבור הכל יחד עם ה-system prompt
            full_prompt = f"""{system_prompt}{chr(10).join(messages)}"""
            
            prompt_tokens = len(tokenizer.encode(full_prompt))
            max_new_tokens = min(request.max_length, max_model_length - prompt_tokens - 50)
            
            logger.info(f"Prompt tokens: {prompt_tokens}")
            logger.info(f"Maximum new tokens allowed: {max_new_tokens}")
                
            encoded = tokenizer(full_prompt, return_tensors='pt').to(model.device)
            
            for attempt in range(3):  # מקסימום 3 ניסיונות
                generated_text = tokenizer.batch_decode(
                    model.generate(
                        **encoded,                        
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,                        
                        do_sample=True,
                        num_beams=1,  # ביטול beam search לשיפור מהירות                        
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        min_length=request.min_length,
                        length_penalty=request.length_penalty
                    )
                )
                
                result = clean_response(generated_text[0] if generated_text else "")
                
                if validate_response(result, request.input_text):
                    break
                    
                logger.warning(f"Invalid response on attempt {attempt + 1}, retrying...")
                
                # הקטנת temperature בכל ניסיון
                request.temperature = max(0.3, request.temperature - 0.2)
        else:
                raise HTTPException(status_code=500, detail="Failed to generate valid response after multiple attempts")
        
        if conversation_id:
            conversation_manager.add_message(conversation_id, "user", request.input_text)
            conversation_manager.add_message(conversation_id, "assistant", result)
            
        return {
            "generated_text": result,
            "token_info": {
                "prompt_tokens": prompt_tokens,
                "max_model_tokens": max_model_length,
                "max_new_tokens_allowed": max_new_tokens,
                "conversation_history_length": len(conversation_context.split('\n')) if conversation_context else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    conversation_manager.clear_conversation(conversation_id)
    return {"status": "conversation cleared"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": MODEL_NAME}
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
    def __init__(self, max_history: int = 5, max_age_minutes: int = 30):
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
MODEL_NAME = "dicta-il/dictalm2.0"
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
    max_length: Optional[int] = Field(default=100, ge=10, le=32768)
    temperature: Optional[float] = Field(default=0.3, ge=0.1, le=1.0)
    language: Optional[str] = Field(default="he", pattern="^(he|en)$")
    min_length: int = Field(default=10, ge=1, le=100)
    length_penalty: float = Field(default=1.0, ge=0.1, le=2.0)

@app.post("/generate/")
async def generate_text(
    request: TextGenerationRequest,
    conversation_id: Optional[str] = Header(None, description="Unique conversation identifier")):

    try:
        max_model_length = model.config.max_position_embeddings
        logger.info(f"Model maximum context length: {max_model_length}")
        
        # במקום הקוד הקיים לקבלת קונטקסט, נשתמש בפונקציה החדשה
        conversation_context = get_clean_conversation_context(
            conversation_id,
            max_tokens=max_model_length // 2
        )
        
        system_prompt = """דבר בטבעיות והשב תשובות קצרות וממוקדות.

        """

        full_prompt = f"{system_prompt}משתמש: {request.input_text.strip()}\nתשובה: "
        
        prompt_tokens = len(tokenizer.encode(full_prompt))
        max_new_tokens = min(request.max_length, max_model_length - prompt_tokens - 50)  # משאיר מרווח בטחון של 50 טוקנים
        
        logger.info(f"Prompt tokens: {prompt_tokens}")
        logger.info(f"Maximum new tokens allowed: {max_new_tokens}")
            
        encoded = tokenizer(full_prompt, return_tensors='pt').to(model.device)
        generated_text = tokenizer.batch_decode(
            model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_k=40,
                top_p=0.9,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                min_length=request.min_length,
                length_penalty=request.length_penalty,
                # מחקנו את early_stopping או לחלופין נוסיף beam search
                num_beams=4,  # אם אתה רוצה להשתמש ב-beam search
                early_stopping=True
            )
        )
        
        # ניקוי התגובה
        result = clean_response(generated_text[0] if generated_text else "")
        result = result.replace("<s>", "").replace("</s>", "").strip()
        result = result.replace(full_prompt, "").strip()
        result = result.replace(system_prompt, "").strip()
        
        # בדיקת קטיעת תשובה
        if result and not any(result.endswith(p) for p in ['.', '!', '?', ':', ';']):
            logger.warning("Response appears truncated, retrying with shorter length")
            return await generate_text(TextGenerationRequest(
                input_text=request.input_text,
                max_length=int(max_new_tokens * 0.8),
                temperature=request.temperature,
                language=request.language,
                min_length=request.min_length,
                length_penalty=request.length_penalty
            ), conversation_id)
        
        # שמירת ההודעות בהיסטוריה
        if conversation_id:
            conversation_manager.add_message(conversation_id, "user", request.input_text)
            conversation_manager.add_message(conversation_id, "assistant", result)
            
        # if not validate_response(result, request.input_text):
        #     logger.warning("Generated invalid response, retrying...")
        #     return await generate_text(request, conversation_id)
        
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

def clean_response(text: str) -> str:
    # הסרת תגיות
    text = text.replace("<s>", "").replace("</s>", "").strip()
    
    # הסרת כל הפרומפטים וההוראות
    patterns_to_remove = [
        r'אתה עוזר אישי.*?שאלה:',  # מסיר את כל ההוראות למודל
        r'היסטוריית שיחה.*?:',      # מסיר את כותרת ההיסטוריה
        r'משתמש:.*?עוזר:',          # מסיר דיאלוגים
        r'שאלה:.*?תשובה:',          # מסיר פורמט שאלה-תשובה
        r'סייען:',                   # מסיר תגית סייען
        r'עוזר:'                     # מסיר תגית עוזר
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # ניקוי רווחים מיותרים
    text = ' '.join(text.split())
    
    return text

def get_clean_conversation_context(conversation_id: str, max_tokens: int) -> str:
    if not conversation_id:
        return ""
        
    context = conversation_manager.get_conversation_context(
        conversation_id,
        max_tokens=max_tokens,
        tokenizer=tokenizer
    )
    
    # וידוא שההיסטוריה נקייה ורלוונטית
    if context:
        context = "היסטוריית שיחה קודמת לידע כללי בלבד:\n" + context
        
    return context

def validate_response(response: str, question: str) -> bool:
    """בדיקה בסיסית של תקינות התשובה"""
    # בדיקה שהתשובה לא ריקה
    if not response or len(response.strip()) < 2:
        return False
        
    # בדיקה שלא קיבלנו רק את הפרומפט בחזרה
    if response.strip().startswith("אתה עוזר אישי"):
        return False
        
    # בדיקה שאין דיאלוג מלא
    dialog_patterns = [
        "משתמש: ",
        "user: ",
        "assistant: "
    ]
    
    if any(pattern in response.lower() for pattern in dialog_patterns):
        return False
        
    return True
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoTokenizer, AutoModelForCausalLM
# import torch
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import os
# import logging

# load_dotenv()

# HF_TOKEN = os.getenv('HF_TOKEN')
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# origins = [
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # טעינת המודל וה-tokenizer
# MODEL_NAME  = "dicta-il/dictalm2.0"
# # Initialize model and tokenizer
# model = AutoModelForCausalLM.from_pretrained('dicta-il/dictalm2.0', torch_dtype=torch.bfloat16, device_map='cuda', load_in_4bit=True, token=HF_TOKEN)
# tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictalm2.0', token=HF_TOKEN)
# # model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# # tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# # Configure tokenizer
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = model.config.eos_token_id

# class TextGenerationRequest(BaseModel):
#     input_text: str
#     max_length: int = Field(default=100, ge=10, le=2000)
#     temperature: float = Field(default=0.3, ge=0.1, le=1.0)
#     language: str = Field(default="he", pattern="^(he|en)$")
#     min_length: int = Field(default=10, ge=1, le=100)
#     length_penalty: float = Field(default=1.0, ge=0.1, le=2.0)

# @app.post("/generate/")
# async def generate_text(request: TextGenerationRequest):
#     try:
#         # בדיקת מגבלות המודל
#         max_model_length = model.config.max_position_embeddings
#         logger.info(f"Model maximum context length: {max_model_length}")
        
#         system_prompt = """אתה עוזר אישי בעברית שמנהל שיחה טבעית. 
# עליך להגיב בצורה רלוונטית והגיונית להודעות של המשתמש.
# יש לוודא שהתשובות שלך מלאות ולא נקטעות באמצע.

# משתמש: """

#         full_prompt = f"{system_prompt}{request.input_text.strip()}\nעוזר: "
        
#         # בדיקת אורך הפרומפט בטוקנים
#         prompt_tokens = len(tokenizer.encode(full_prompt))
#         max_new_tokens = min(request.max_length, max_model_length - prompt_tokens)
        
#         logger.info(f"Prompt tokens: {prompt_tokens}")
#         logger.info(f"Maximum new tokens allowed: {max_new_tokens}")
            
#         encoded = tokenizer(full_prompt, return_tensors='pt').to(model.device)
#         generated_text = tokenizer.batch_decode(
#             model.generate(
#                 **encoded,
#                 do_sample=True,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.3,
#                 top_k=40,
#                 top_p=0.9,
#                 no_repeat_ngram_size=3,
#                 repetition_penalty=1.3,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,  # וידוא שהמודל יודע מתי לסיים
#                 min_length=10,  # מינימום אורך לתשובה
#                 length_penalty=1.0,  # עידוד תשובות ארוכות יותר
#                 early_stopping=True  # עצירה רק כשמגיעים לסוף טבעי
#             )
#         )
        
#         # ניקוי התגובה
#         result = generated_text[0] if generated_text else ""
#         result = result.replace("<s>", "").replace("</s>", "").strip()
#         result = result.replace(full_prompt, "").strip()
#         result = result.replace(system_prompt, "").strip()
        
#         # בדיקה שהתשובה לא נקטעה באמצע משפט
#         if result and not any(result.endswith(p) for p in ['.', '!', '?', ':', ';']):
#             # אם התשובה נקטעה, נסה שוב עם פחות טוקנים
#             logger.warning("Response appears truncated, retrying with shorter length")
#             return await generate_text(TextGenerationRequest(
#                 input_text=request.input_text,
#                 max_length=int(max_new_tokens * 0.8),  # 80% מהאורך המקורי
#                 temperature=request.temperature,
#                 language=request.language
#             ))
            
#         return {
#             "generated_text": result,
#             "token_info": {
#                 "prompt_tokens": prompt_tokens,
#                 "max_model_tokens": max_model_length,
#                 "max_new_tokens_allowed": max_new_tokens
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Error generating text: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/generate/")
# # async def generate_text(request: TextGenerationRequest):
# #     try:
# #         logger.info(f"Received request with text: {request.input_text}")
        
# #         if not request.input_text.strip():
# #             raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
# #         # יצירת פרומפט עם הקשר ברור והנחיות
# #         system_prompt = """אתה עוזר אישי בעברית שמנהל שיחה טבעית. 
# #         עליך להגיב בצורה רלוונטית והגיונית להודעות של המשתמש.
# #         אם המשתמש אומר שלום או היי, הגב בברכה מתאימה.
# #         הקפד תמיד להתייחס למה שהמשתמש כתב.
        
# #         לדוגמה:
# #         משתמש: היי
# #         עוזר: היי! מה שלומך? איך אני יכול לעזור?

# #         משתמש: שלום
# #         עוזר: שלום וברוכים הבאים! במה אוכל לסייע לך?

# #         משתמש: """

# #         full_prompt = f"{system_prompt}{request.input_text.strip()}\nעוזר: "
            
# #         encoded = tokenizer(full_prompt, return_tensors='pt').to(model.device)
# #         generated_text = tokenizer.batch_decode(
# #             model.generate(
# #                 **encoded,
# #                 do_sample=False,
# #                 max_new_tokens=request.max_length,
# #                 temperature=0.3,  # הורדתי את הטמפרטורה לתשובות יותר ממוקדות
# #                 num_beams=5,
# #                 top_k=40,
# #                 top_p=0.9,
# #                 no_repeat_ngram_size=3,
# #                 repetition_penalty=1.3,
# #                 pad_token_id=tokenizer.eos_token_id,
# #                 early_stopping=True
# #             )
# #         )
        
# #         # ניקוי התגובה
# #         result = generated_text[0] if generated_text else ""
# #         result = result.replace("<s>", "").replace("</s>", "").strip()
        
# #         # הסרת הפרומפט המקורי מהתשובה
# #         result = result.replace(full_prompt, "").strip()
# #         result = result.replace(system_prompt, "").strip()
        
# #         # וידוא שיש תוכן בתשובה
# #         if not result:
# #             result = "שלום! איך אני יכול לעזור לך?"
            
# #         return {"generated_text": result}

# #         # input_ids = tokenizer(request.input_text, return_tensors="pt").to("cuda")
# #         # outputs = model.generate(**input_ids)
# #         # print(tokenizer.decode(outputs[0]))

# #         # prompt = f"Question: {request.text}\nAnswer: "
# #         # inputs = tokenizer.encode_plus(
# #         #     prompt,
# #         #     return_tensors="pt",
# #         #     add_special_tokens=True,
# #         #     padding=True,
# #         #     truncation=True,
# #         #     max_length=512,
# #         #     return_attention_mask=True
# #         # )
        
# #         # outputs = model.generate(
# #         #     inputs["input_ids"],
# #         #     attention_mask=inputs["attention_mask"],
# #         #     max_length=request.max_length,
# #         #     temperature=request.temperature,
# #         #     top_k=50,
# #         #     top_p=0.95,
# #         #     no_repeat_ngram_size=2,
# #         #     num_return_sequences=1,
# #         #     do_sample=True,
# #         #     pad_token_id=tokenizer.pad_token_id,
# #         #     repetition_penalty=1.2,
# #         # )
        
# #         # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
# #         # Clean up the generated text
# #         # if "." in generated_text:
# #         #     generated_text = ". ".join(generated_text.split(".")[:-1]) + "."
        
# #         # return {"generated_text": generated_text}
# #     except Exception as e:
# #         logger.error(f"Error generating text: {str(e)}")
# #         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     return {"status": "ok", "model": MODEL_NAME}
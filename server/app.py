import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "dicta-il/dictalm2.0-instruct"

# Load model + tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_model_reply(hidden_history):
    """
    Receives the entire conversation in hidden_history (list of {'role': str, 'content': str}).
    Passes it all to the model for context, and returns the model's new reply.
    """
    # Convert the list of messages to a single prompt
    conversation_text = ""
    for msg in hidden_history:
        if msg["role"] == "user":
            conversation_text += f"User: {msg['content']}\n"
        else:
            conversation_text += f"Assistant: {msg['content']}\n"

    inputs = tokenizer(conversation_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        pad_token_id=model.config.eos_token_id
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Optional: remove the user's text if it was repeated
    # (some models repeat the user input in the answer)
    return reply

def user_asks(user_input, hidden_history):
    """
    1. Append the user's new message to hidden_history (model context).
    2. Generate a new model reply using the entire hidden_history.
    3. Append that model reply to hidden_history.
    4. Return only the last two messages (latest user + assistant) to display,
       while returning the full hidden_history in the background.
    """
    if hidden_history is None:
        hidden_history = []

    # 1. add the user's message
    hidden_history.append({"role": "user", "content": user_input})
    
    # 2. generate the model reply
    model_reply = generate_model_reply(hidden_history)
    
    # 3. add the assistant message
    hidden_history.append({"role": "assistant", "content": model_reply})
    
    # 4. build a smaller list to show only the latest exchange
    displayed_messages = hidden_history[-2:]
    
    return displayed_messages, hidden_history

with gr.Blocks(css='''
  .gr-chatbot { direction: rtl; text-align: right; }
  .gr-textbox textarea { direction: rtl; text-align: right; }
  .gr-markdown h1, .gr-markdown p { text-align: right; direction: rtl; }
  .gr-input, .gr-output { direction: rtl; text-align: right; }
''') as demo:
    gr.Markdown("<h1 style='text-align:right'>Chat with hidden history</h1>")
    
    chatbot = gr.Chatbot(
        label="Chatbot",
        type="messages"  # IMPORTANT: removes the tuples deprecation warning
    )
    state = gr.State([])  # hidden_history
    
    txt = gr.Textbox(
        label="Enter your question",
        placeholder="What do you want to ask?"
    )
    
    # On submit: run user_asks(...),
    # which returns (displayed_messages, hidden_history)
    txt.submit(
        fn=user_asks,
        inputs=[txt, state],
        outputs=[chatbot, state],
        scroll_to_output=True
    )

demo.launch()

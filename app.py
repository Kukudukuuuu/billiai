import torch
import pickle
import gradio as gr
from billiai import BilliAI, CharTokenizer, CONFIG

# ── Load model once at startup ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    checkpoint = torch.load("billiai.pt", map_location=DEVICE)
    saved_cfg  = checkpoint["config"]
    tokenizer  = CharTokenizer.load(saved_cfg["tokenizer_file"])
    saved_cfg["vocab_size"] = tokenizer.vocab_size
    model = BilliAI(saved_cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()
print(f"Billi AI loaded on {DEVICE} ✅")

# ── Generation function ──
def chat(message, history):
    prompt = f"Human: {message}\nBilli AI:"

    context = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(context, max_new_tokens=80, temperature=0.8, top_k=40)

    decoded = tokenizer.decode(out[0].tolist())
    reply = decoded[len(prompt):]

    # Stop at next "Human:" turn
    if "Human:" in reply:
        reply = reply[:reply.index("Human:")]

    return reply.strip()

# ── Gradio UI ──
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 Billi AI",
    description="A tiny language model built from scratch. Say hi!",
    examples=["hi", "what is your name?", "tell me a joke", "how are you?"],
)

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

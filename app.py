import streamlit as st
import torch
import random

from src.model import GPTModel
from src.tokenizer import WordTokenizer
from configs.config import *
from data.prepare_data import load_data
from src.utils import generate_causal_mask

# Load data
text = load_data("data/input.txt")
tokenizer = WordTokenizer(text)

# 🔥 Build lookup table (IMPORTANT)
pairs = {}
lines = text.split("\n")

for line in lines:
    if "=>" in line:
        parts = line.split("=>")
        if len(parts) == 2:
            inp = parts[0].strip().lower()
            out = parts[1].strip()

            if inp not in pairs:
                pairs[inp] = []
            pairs[inp].append(out)

# Load model
model = GPTModel(
    tokenizer.vocab_size,
    embed_dim,
    num_heads,
    num_layers,
    ff_dim,
    max_len
).to(device)

model.load_state_dict(torch.load("outputs/checkpoints/model.pt"))
model.eval()

st.title("Mini Chatbot 🤖")

prompt = st.text_input("Type your message:")

if st.button("Send"):

    user_input = prompt.lower().strip()

    # ✅ 1. Exact match (MAIN FIX)
    if user_input in pairs:
        output = random.choice(pairs[user_input])

    else:
        # ❗ fallback to model (for unknown inputs)
        tokens = tokenizer.encode(user_input + " =>")

        for _ in range(50):
            x = torch.tensor(tokens).unsqueeze(0).to(device)
            mask = generate_causal_mask(x.size(1)).to(device)

            logits = model(x, mask)

            probs = torch.softmax(logits[0, -1] / 1.2, dim=-1)

            # repetition penalty
            for t in set(tokens[-10:]):
                probs[t] *= 0.5

            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs / top_probs.sum()

            next_token = top_indices[torch.multinomial(top_probs, 1)].item()
            tokens.append(next_token)

            decoded = tokenizer.decode(tokens)

            # stop early (important)
            if len(decoded.split()) > 6:
                break

        output = tokenizer.decode(tokens)

        if "=>" in output:
            output = output.split("=>")[-1]

        output = output.strip()

        if output == "":
            output = "I am not sure."

    st.write(output)
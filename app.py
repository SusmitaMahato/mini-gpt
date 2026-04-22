import streamlit as st
import torch

from src.model import GPTModel
from src.tokenizer import WordTokenizer
from configs.config import *
from data.prepare_data import load_data
from src.utils import generate_causal_mask

# Load data
text = load_data("data/input.txt")
tokenizer = WordTokenizer(text)

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
    tokens = tokenizer.encode("User: " + prompt + " Bot:")

    for _ in range(100):
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        mask = generate_causal_mask(x.size(1)).to(device)

        logits = model(x, mask)

        probs = torch.softmax(logits[0, -1] / 0.6, dim=-1)

        for t in set(tokens[-30:]):
            probs[t] *= 0.1

        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs / top_probs.sum()

        next_token = top_indices[torch.multinomial(top_probs, 1)].item()
        tokens.append(next_token)

        # stop condition
        decoded = tokenizer.decode(tokens)
        if len(tokens) > 40 and decoded.count(".") >= 2:
            break

    output = tokenizer.decode(tokens)
    st.write(output)
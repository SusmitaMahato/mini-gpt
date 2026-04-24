import torch
from configs.config import * 
from src.model import GPTModel 
from src.tokenizer import WordTokenizer 
from src.utils import generate_causal_mask 
from data.prepare_data import load_data

def generate():
    # Load data for tokenizer
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

    # Prompt
    prompt = input("Enter prompt: ")
    tokens = tokenizer.encode("User: " + prompt + " Bot:")

    for _ in range(100):
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        mask = generate_causal_mask(x.size(1)).to(device)

        logits = model(x, mask)

        probs = torch.softmax(logits[0, -1] / 0.6, dim=-1)

        # repetition penalty
        for t in set(tokens[-30:]):
            probs[t] *= 0.1

        # top-k sampling
        top_k = 7
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs / top_probs.sum()

        next_token = top_indices[torch.multinomial(top_probs, 1)].item()

        tokens.append(next_token)

        # 🔥 STOP EARLY if model starts new User
        last_word = tokenizer.decode([next_token])
        if last_word == "User:":
            break

        # 🔥 length + sentence stopping
        decoded = tokenizer.decode(tokens)

        if len(tokens) > 40 and decoded.count(".") >= 2:
            break

    # 🔥 CLEAN OUTPUT
    output_text = tokenizer.decode(tokens)

    if "Bot:" in output_text:
        output_text = output_text.split("Bot:")[-1]

    if "User:" in output_text:
        output_text = output_text.split("User:")[0]

    output_text = output_text.strip()

    if not output_text.endswith((".", "!", "?")):
        output_text += "."

    print(output_text)

    with open("outputs/samples.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
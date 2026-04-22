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

    # Starting prompt
    prompt = input("Enter prompt: ")
    tokens = tokenizer.encode("User: " + prompt + " Bot:")

    # Generate text
    for _ in range(100):
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        mask = generate_causal_mask(x.size(1)).to(device)

        logits = model(x, mask)

        # FIXED: Use sampling instead of argmax
        probs = torch.softmax(logits[0, -1] / 0.5, dim=-1)

        # reduce repetition
        # repetition penalty
        for t in set(tokens[-30:]):
            probs[t] *= 0.1

        if len(tokens) > 5:
            if tokens[-3:] == tokens[-6:-3]:
                probs[tokens[-1]] *= 0.05   

        top_k = 7
        top_probs, top_indices = torch.topk(probs, top_k)

        # normalize probabilities
        top_probs = top_probs / top_probs.sum()

        # sample from top-k
        next_token = top_indices[torch.multinomial(top_probs, 1)].item()
        next_token = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token)
        decoded = tokenizer.decode(tokens)
        if "User:" in decoded:
            break

        decoded_text = tokenizer.decode(tokens)
                    
        if len(tokens) > 60 and decoded_text.count(".") >= 2:
            break
        
        if len(tokens) > 40:
            last_word = tokenizer.decode([tokens[-1]])

            if last_word in [".", "!", "?"] and len(tokens) > 60:
                break

    output_text = tokenizer.decode(tokens)
    # 🔥 extract only Bot response
    if "Bot:" in output_text:
        output_text = output_text.split("Bot:")[-1]

    # remove next User part if generated
    if "User:" in output_text:
        output_text = output_text.split("User:")[0]

    output_text = output_text.strip()

# ensure clean ending
    if not output_text.endswith((".", "!", "?")):
        output_text += "."

    print(output_text)
    # Save output
    
    with open("outputs/samples.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
if __name__ == "__main__":
    generate()
import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import *
from src.model import GPTModel
from src.utils import generate_causal_mask
from data.prepare_data import load_data, create_dataset, get_batch
from src.tokenizer import WordTokenizer

def train():
    text = load_data("data/input.txt")
    tokenizer = WordTokenizer(text)

    encoded = tokenizer.encode(text)
    train_data, val_data = create_dataset(encoded)

    model = GPTModel(
        tokenizer.vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        max_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        x, y = get_batch(train_data, batch_size)

        x, y = x.to(device), y.to(device)
        mask = generate_causal_mask(x.size(1)).to(device)

        logits = model(x, mask)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss {loss.item()}")

    torch.save(model.state_dict(), "outputs/checkpoints/model.pt")

if __name__ == "__main__":
    train()
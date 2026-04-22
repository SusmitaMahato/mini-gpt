import torch
from configs.config import block_size

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_dataset(encoded_text):
    data = torch.tensor(encoded_text, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y
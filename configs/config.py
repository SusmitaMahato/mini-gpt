# Model
embed_dim = 256
num_heads = 8
num_layers = 6
ff_dim = 512
block_size = 128
max_len = 256

# Training
batch_size = 4
lr = 3e-4
epochs = 300

# Device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
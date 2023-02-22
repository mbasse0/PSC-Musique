import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from vocab import custom_vocab

# Define hyperparameters
batch_size = 32
num_epochs = 5
num_tokens = len(custom_vocab)
embed_dim = len(custom_vocab) #la taille du voc car on fait du one hot
num_heads = 283
hidden_dim = 64
dropout = 0.1

# Define the Transformer model
class TransformerModel(torch.nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.decoder = torch.nn.Linear(embed_dim, num_tokens)
        self.init_weights()
        
    def forward(self, x, memory):
        x = x.to(torch.float32)
        x = self.transformer_decoder(x, memory)
        x = self.decoder(x)
        return x
        
    def init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        
# Define the dataset
class MyDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.input_data[idx], dtype=torch.long)
        output_tensor = torch.tensor(self.output_data[idx], dtype=torch.long)
        return input_tensor, output_tensor


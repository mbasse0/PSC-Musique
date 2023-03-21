import torch
from torch.utils.data import Dataset, DataLoader
from vocab import *


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


def get_dataloader(input_vect, rep_vect, batch_size):
    # Create the dataset
    dataset = MyDataset(input_vect, rep_vect)
    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
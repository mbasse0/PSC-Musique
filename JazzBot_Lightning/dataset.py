import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
#from vocab import *


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    return dataloader

def get_two_dataloaders(input_vect, rep_vect, batch_size):
    # Create the dataset
    dataset = MyDataset(input_vect, rep_vect)
    
    # use 20% of training data for validation
    n = len(input_vect)
    train_set_size = int(n * 0.9)
    valid_set_size = n - train_set_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    
    # Create a dataloader
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=12)
    return train_dataloader,val_dataloader
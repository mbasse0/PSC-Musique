from data_processor import *
from model import MyDataset
from torch.utils.data import DataLoader

def dataloader(folder_path,
               batch_size,
               N):
    '''
    Args : N = number of notes we want in the pieces
    '''
    input, target = folderToVectInputTarget(folder_path,N)
    dataset =  MyDataset(input, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


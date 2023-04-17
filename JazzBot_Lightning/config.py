import torch 

nombre_ordis = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = "../Data/"
N = 25 #nb_note

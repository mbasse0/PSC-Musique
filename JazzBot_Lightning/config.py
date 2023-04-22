import torch 

nombre_ordis = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N = 120 #nb token entraînement

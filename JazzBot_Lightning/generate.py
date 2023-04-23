from config import *
from model import *
from vocab import *

import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"



def generate_sequence(model, start_tokens, max_length=100, temperature=1.0, progress_callback=None):
    # start_tokens doit être une liste d'indices
    model.eval()

    # L'input donné à l'encoder (vecteur nul dans notre cas, comme pendant l'entraînement)
    taille_bloc = 120
    X = torch.tensor([0]*taille_bloc,device=device).unsqueeze(0).to(device)
    N = max_length - len(start_tokens)
    with torch.no_grad():
        les_tokens = start_tokens
        les_genere = []
        for i in (range(max_length - len(start_tokens))):
            # Unsqueeze(0) rajoute une dimension qui correspond au batch_size (qui vaut 1 dans ce cas) pour coller aux shape attendues par le modèle
            input_tokens = torch.tensor(start_tokens, device=device).unsqueeze(0).to(device)
            
            output = model(X, input_tokens)
            # Les logits sont des probabilits non normalisées. La température contrôle leur dispersion : permet d'ajouter plus ou moins de bruit lors de la prédiction
            logits = output[:, -1, :] / temperature
            # Softmax transforme les logits en probabilités, multinomial fait une séleciton pondérée par ces probabilités d''un seul indice (num_samples=1), to_list passe de tensor à array
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze().tolist()
            les_tokens.append(next_token)
            les_genere.append(next_token)
            if progress_callback:
            
                progress_callback(i / N)

    # print([itos_vocab[el] for el in les_genere])
    return les_tokens



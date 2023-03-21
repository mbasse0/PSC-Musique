from config import *
from model import *
from data_processor import *

import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# # On charge le modèle pré entraîné
# model = Transformer(
# num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1)
# model.load_state_dict(torch.load("model4out.pth"))
# # On déplace le modèle sur le GPU si c'est possible
# model.to(device)
# # On met le modèle en mode inférence pour pas avoir le dropout et obtenir des prédictions plus précises
# model.eval()





def generate_sequence(model, start_tokens, max_length=100, temperature=1.0):
    model.eval()

    # L'input donné à l'encoder (vecteur nul dans notre cas, comme pendant l'entraînement)
    taille_bloc = 120
    X = torch.tensor([0]*taille_bloc).unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = start_tokens
        for _ in range(max_length - len(start_tokens)):
            # Unsqueeze(0) rajoute une dimension qui correspond au batch_size (qui vaut 1 dans ce cas) pour coller aux shape attendues par le modèle
            input_tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            
            output = model(X, input_tokens)
            # Les logits sont des probabilits non normalisées. La température contrôle leur dispersion : permet d'ajouter plus ou moins de bruit lors de la prédiction
            logits = output[:, -1, :] / temperature
            # Softmax transforme les logits en probabilités, multinomial fait une séleciton pondérée par ces probabilités d''un seul indice (num_samples=1), to_list passe de tensor à array
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze().tolist()
            tokens.append(next_token)

    return tokens



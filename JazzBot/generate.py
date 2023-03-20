from config import *
from model import *
from data_processor import *

import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# On charge le modèle pré entraîné
model = Transformer(
num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1)
model.load_state_dict(torch.load("model4out.pth"))
# On déplace le modèle sur le GPU si c'est possible
model.to(device)
# On met le modèle en mode inférence pour pas avoir le dropout et obtenir des prédictions plus précises
model.eval()


# L'input donné à l'encoder (vecteur nul dans notre cas, comme pendant l'entraînement)
taille_bloc = 120
X = torch.tensor([0]*taille_bloc).unsqueeze(0).to(device)


def generate_sequence(model, start_tokens, max_length=100, temperature=1.0):
    model.eval()

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


# Define the start tokens for your inference. start_tokens expects an array of indices in the vocab
start_tokens = [custom_vocab["n60"], custom_vocab["d1"], custom_vocab["t1"], custom_vocab["v64"]]
#start_tokens = ['d1', 't1', 'v109', 'n61', 'd1', 't1', 'v105', 'n65', 'd1', 't4', 'v105', 'n70', 'd5', 't2', 'v114', 'n67', 'd1', 't6', 'v104', 'n63', 'd1', 't1', 'v104', 'n60', 'd1', 't0', 'v105', 'n61', 'd2', 't1', 'v105', 'n68', 'd1', 't2', 'v103', 'n65', 'd1', 't16', 'v114', 'n64', 'd4', 't1', 'v105', 'n67', 'd1', 't5', 'v108', 'n63', 'd1', 't2', 'v91', 'n60', 'd1', 't3', 'v95', 'n65', 'd2', 't1', 'v113', 'n62', 'd1', 't2', 'v104', 'n63', 'd1', 't1', 'v111', 'n65', 'd1', 't2', 'v112', 'n61', 'd2', 't1', 'v101', 'n62', 'd1', 't2', 'v99', 'n63', 'd2', 't1', 'v111', 'n61', 'd1', 't2', 'v105', 'n62', 'd2', 't1', 'v114', 'n58', 'd1', 't2', 'v114', 'n53', 'd2', 't1', 'v106', 'n51', 'd7', 't2', 'v107', 'n58', 'd1', 't7', 'v112', 'n51', 'd2', 't2', 'v108', 'n52', 'd4', 't2', 'v110', 'n53', 'd7', 't4', 'v109', 'n53']

#start_tokens = [custom_vocab[el] for el in start_tokens]
#start_tokens = np.load("morceau_RAS.npy").tolist()
# Generate a sequence of tokens
generated_tokens = generate_sequence(model, start_tokens, max_length=100, temperature=1.0)

# Decode the generated tokens into the original format
decoded_tokens = [itos_vocab[el] for el in generated_tokens]
#On peut sauver la prédiction dans un array
#np.save("generation4.npy", decoded_tokens)
print("Generated sequence:", decoded_tokens)
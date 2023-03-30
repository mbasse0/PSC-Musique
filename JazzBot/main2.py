from vocab import *
from generate import *
from data_encoder import *
from data_decoder import *
from train import *
from dataset import *


## ENCODING DATA

# les_tokens = midifolderToTokens("midi_data")
# #répartir le data du morceau en blocs de 120 attributs (30 notes)
# #Et associer à chaque bloc la réponse attendue (l'attribut suivant)
# taille_bloc = 120
# les_morceaux = []
# les_morceaux_rep = []

# for i in range(len(les_tokens)//(taille_bloc+1)-1):
#     les_morceaux.append(les_tokens[i:i+taille_bloc-1])
#     les_morceaux_rep.append(les_tokens[i:i+taille_bloc])
input_vect = [ [0] + [ custom_vocab[tok] for tok in morceau] for morceau in les_morceaux ]
rep_vect = [ [ custom_vocab[tok] for tok in morceau] for morceau in les_morceaux_rep ]


# CREATION DATASET ET DATALOADER

batch_size = 32
dataloader = get_dataloader(input_vect, rep_vect, batch_size)

for batch in dataloader:
    y1, y2 = batch
    print("size y1", y1.shape)

## ENTRAINEMENT

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1
).to(device)

model.load_state_dict(torch.load("model4out.pth"))


# opt = torch.optim.SGD(model.parameters(), lr=0.01)
# loss_fn = nn.CrossEntropyLoss()

# epochs = 10
# train_loss_list = fit(model, opt, loss_fn, dataloader, epochs)
# # save the model weights to a file
# torch.save(model.state_dict(), 'model4out.pth')


##GENERER SEQ

# Define the start tokens for your inference. start_tokens expects an array of indices in the vocab
start_tokens = [custom_vocab["n60"], custom_vocab["d1"], custom_vocab["t1"], custom_vocab["v64"]]
# Generate a sequence of tokens
generated_tokens = generate_sequence(model, start_tokens, max_length=100, temperature=1.0)

# Decode the generated tokens into the original format
decoded_tokens = [itos_vocab[el] for el in generated_tokens]
#On peut sauver la prédiction dans un array
#np.save("generation4.npy", decoded_tokens)
print("Generated sequence:", decoded_tokens)


## CONVERSION DE LA SEQUENCE EN MIDI

# tokens_to_midi(decoded_tokens, "midi4.mid", 120)
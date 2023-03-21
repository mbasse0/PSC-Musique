from vocab import *
from generate import *
from data_encoder import *
from data_decoder import *
from train import *
from dataset import *

## ENCODING DATA

les_tokens = midifolderToTokens("midi_data")


#répartir le data du morceau en blocs de 120 attributs (30 notes)
#Et associer à chaque bloc la réponse attendue (l'attribut suivant)

taille_bloc = 120
les_morceaux = []
les_morceaux_rep = []

for i in range(len(les_tokens)//(taille_bloc+1)-1):
    les_morceaux.append(les_tokens[i:i+taille_bloc-1])
    les_morceaux_rep.append(les_tokens[i:i+taille_bloc])
input_vect = [ [0] + [ custom_vocab[tok] for tok in morceau] for morceau in les_morceaux ]
rep_vect = [ [ custom_vocab[tok] for tok in morceau] for morceau in les_morceaux_rep ]

## CREATION DATASET ET DATALOADER
batch_size = 32
dataloader = get_dataloader(input_vect, rep_vect, batch_size)

## ENTRAINEMENT

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1
).to(device)

opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


train_loss_list = fit(model, opt, loss_fn, dataloader, 1)


# save the model weights to a file
torch.save(model.state_dict(), 'model4out.pth')


##GENERER SEQ



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
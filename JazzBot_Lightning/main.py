from vocab import *
from model import *
import numpy as np
from generate import *
from data_encoder import *
from data_decoder import *
##from train import *
from dataset import *
import pytorch_lightning as pl
from config import *

if __name__ == '__main__':
   ## ENCODING DATA

   # # les_tokens = midifolderToTokens("midi_data")
   # # #répartir le data du morceau en blocs de 120 attributs (30 notes)
   # # #Et associer à chaque bloc la réponse attendue (l'attribut suivant)
   # # taille_bloc = 120
   # # les_morceaux = []
   # # les_morceaux_rep = []

   # # for i in range(len(les_tokens)//(taille_bloc+1)-1):
   # #     les_morceaux.append(les_tokens[i:i+taille_bloc-1])
   # #     les_morceaux_rep.append(les_tokens[i:i+taille_bloc])
   # # input_vect = [ [0] + [ custom_vocab[tok] for tok in morceau] for morceau in les_morceaux ]
   # # rep_vect = [ [ custom_vocab[tok] for tok in morceau] for morceau in les_morceaux_rep ]

   # # np.save('input_dataset2.npy', input_vect)
   # # np.save('rep_dataset2.npy', rep_vect)

   #input_vect = np.load('input_weimar.npy')
   #rep_vect = np.load('rep_weimar.npy')
   
   # Weimar120
   input_vect, rep_vect = tokensFileToVectInputTarget("WeimarFinal.csv",120)

   np.save('input_weimar120.npy', input_vect)
   np.save('rep_weimar120.npy', rep_vect)


   # input_vect = np.load('input_weimar120.npy')
   # rep_vect = np.load('rep_weimar120.npy')

   # ## CREATION DATASET ET DATALOADER

   batch_size = 32
   dataloader = get_dataloader(input_vect, rep_vect, batch_size)
   train_dataloader, val_dataloader = get_two_dataloaders(input_vect, rep_vect, batch_size)


   ## ENTRAINEMENT

   #model = Transformer4(
   #   n_toks = len(itos_NOTE), d_toks = len(itos_DUR), t_toks = len(itos_TIM), v_toks = len(itos_VEL),
   #   dim_model=256, num_heads=8, num_encoder_layers=1, num_decoder_layers=6, dropout_p=0.1
   #)

   model = Transformer(
   num_tokens=len(custom_vocab), dim_model=256, num_heads=8, num_encoder_layers=1, num_decoder_layers=6, dropout_p=0.1
   )

   logger = pl.loggers.TensorBoardLogger(save_dir='.')
   trainer = pl.Trainer(accelerator='gpu', max_epochs=1, log_every_n_steps=20, benchmark=True, logger=logger)
   #trainer = pl.Trainer(accelerator='gpu', gpus=3, strategy='ddp', max_epochs=1, log_every_n_steps=20, benchmark=True, profiler="simple", logger=logger)
   trainer.fit(model, train_dataloader, val_dataloader)

   # save the model weights to a file
   torch.save(model.state_dict(), 'test.pth')


   # ##GENERER SEQ

   # # Define the start tokens for your inference. start_tokens expects an array of indices in the vocab
   # start_tokens = [custom_vocab["n60"], custom_vocab["d1"], custom_vocab["t1"], custom_vocab["v64"]]
   # # Generate a sequence of tokens
   # model.load_state_dict(torch.load('model4out_rect.pth'))
   # generated_tokens = generate_sequence(model, start_tokens, max_length=300, temperature=1)

   # # Decode the generated tokens into the original format
   # decoded_tokens = [itos_vocab[el] for el in generated_tokens]
   # #On peut sauver la prédiction dans un array
   # #np.save("generation4.npy", decoded_tokens)
   # print("Generated sequence:", decoded_tokens)


   # ## CONVERSION DE LA SEQUENCE EN MIDI

   # tokens_to_midi(decoded_tokens, "midi3.mid", 100)
   # # tokens_to_midi([itos_vocab[el]for el in input_vect[10]], "midi_dataset.mid", 120)
   # # tokens_to_midi([itos_vocab[el]for el in input_vect[10]], "midi_dataset_GM.mid", 120)
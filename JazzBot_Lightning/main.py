from vocab import *
from model import *
from model4out import *
from generate import *
from data_decoder import *
from data_encoder import *
from dataset import *
from csv_encoder import *
from config import *

import pytorch_lightning as pl
import numpy as np
import sys

def main(argv):
   """
   arg0 : "train" : entraîner , "gen" : génére, "fine_tune" : fine tuner
   arg1 : model_name (to create if "train" and to fine tune if "fine_tune")
   arg2 : 0 : model , 1 : model4out

   (arguments uniquement valables pour "train")
   arg3 : 0 : noDDP , 1 : DDP
   arg4 : nb_epochs
   arg5 : dataset_name

   (arguments uniquement valables pour "gen")
   arg3 : nb_generated_tokens
   arg4 : temperature
   arg5 : start_midi_name (optionnel)
   
   """

   if argv[0] == "train" or argv[0] == "fine_tune" :
      ## ENTRAINEMENT avec les parametres optimaux (loss < 0.2 au bout de 20epoch)
      batch_size = 8

      nb_epochs = int(argv[4])

      dataset_path = "./Datasets/" + argv[5]
      input_vect, rep_vect = tokensFileToVectInputTarget(dataset_path,120)
      train_dataloader, val_dataloader = get_two_dataloaders(input_vect, rep_vect, batch_size)
   
      if int(argv[2]) == 1:
         model = Transformer4(
            dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1, learning_rate = 0.05
         )
      else:
         model = Transformer(
            num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1, learning_rate= 0.06
         )

      # for fine tuning, load the pretrained model
      if argv[0] == "fine_tune":
         model.load_state_dict(torch.load("./Models/" + argv[1]))

      logger = pl.loggers.TensorBoardLogger(save_dir='./fine_tune')

      if int(argv[3]) == 1:
         trainer = pl.Trainer(accelerator='gpu', gpus=3, strategy='ddp', max_epochs=nb_epochs, log_every_n_steps=20, benchmark=True, logger=logger)
      else:
         trainer = pl.Trainer(accelerator='gpu', gpus=1, max_epochs=nb_epochs, log_every_n_steps=20, benchmark=True, logger=logger)

      trainer.fit(model, train_dataloader, val_dataloader)

      model_path = "./Models/" + argv[1] + ("" if argv[0] == "train" else "_fine_tune_on" + argv[5])
      torch.save(model.state_dict(), model_path)
   


   elif argv[0] == "gen":
      ## GENERATION
      if len(argv) == 5:
         start_tokens = [custom_vocab["n60"], custom_vocab["d4"], custom_vocab["t1"], custom_vocab["v64"], custom_vocab["n63"], custom_vocab["d1"], custom_vocab["t5"], custom_vocab["v56"], custom_vocab["n65"], custom_vocab["d3"], custom_vocab["t6"], custom_vocab["v64"]]
         temp = float(argv[4])
      else:
         midi_path = "./Midis/" + argv[5]
         start_tokens = midiToTokens(midi_path)
         

      if int(argv[2]) == 1:
         model = Transformer4(
            dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1, learning_rate = 0.05
         )
      else:
         model = Transformer(
            num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1, learning_rate=0.05
         )

      model_path = "./Models/" + argv[1]
      model.load_state_dict(torch.load(model_path, map_location=device))
      model.to(device)

      nb_tokens = int(argv[3])
      generated_tokens = generate_sequence(model, start_tokens, max_length=nb_tokens, temperature = temp)
      decoded_tokens = [itos_vocab[el] for el in generated_tokens]
      print(decoded_tokens)

      # ## CONVERSION DE LA SEQUENCE EN MIDI

      tokens_to_midi(decoded_tokens, "midis/" + argv[1] + "temp" + argv[4] + ".mid", 100)
      # # tokens_to_midi([itos_vocab[el]for el in input_vect[10]], "midi_dataset.mid", 120)
      # # tokens_to_midi([itos_vocab[el]for el in input_vect[10]], "midi_dataset_GM.mid", 120)

if __name__=="__main__":
   main(sys.argv[1:])

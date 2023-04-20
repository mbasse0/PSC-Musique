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
from csv_encoder import *

if __name__ == '__main__':
   #Weimar120
   input_vect, rep_vect = tokensFileToVectInputTarget("WeimarFinal.csv",120)

   np.save('input_weimar120.npy', input_vect)
   np.save('rep_weimar120.npy', rep_vect)
from torchtext.vocab import vocab
from collections import OrderedDict

#version simplifiée de vocab4types

#special tokens
SOS = "sos" #start
EOS = "eos" #end
PAD = "xxx"  #NAN

SPECIAL = [SOS,EOS,PAD]

NOTE_SIZE = 128
DUR_SIZE = 160
TIM_SIZE = 1000
VEL_SIZE = 128

NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
TIM_TOKS = [f't{i}' for i in range(TIM_SIZE)]
VEL_TOKS = [f'v{i}' for i in range(VEL_SIZE)]

# Le token dummy sert seulement à initialiser les mots du vocab à partir de l'index 1, conformément aux prérequis de la fonction vocab()
VOCAB = ["dummy"] + SPECIAL + NOTE_TOKS + DUR_TOKS + SPECIAL + TIM_TOKS + VEL_TOKS 

DICT = [(element, index) for index, element in enumerate(VOCAB)]

CV = vocab(OrderedDict(DICT)) #CV :: custom_vocab

itos= CV.get_itos()

def custom_vocab(tok):
    '''
    tok = [note,dur,tim,vel]
    '''
    return [CV[tok[0]],
            CV[tok[1]],
            CV[tok[2]],
            CV[tok[3]]]

def cv_eos():
    '''
    return token corresponding to eos
    '''
    return [CV[EOS],
            CV[EOS],
            CV[EOS],
            CV[EOS]]

def cv_sos():
    '''
    return token corresponding to sos
    '''
    return [CV[SOS],
            CV[SOS],
            CV[SOS],
            CV[SOS]]
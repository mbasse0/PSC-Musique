from torchtext.vocab import vocab
from collections import OrderedDict

#Vocab


NOTE_SIZE = 128
DUR_SIZE = 160
TIM_SIZE = 1000
VEL_SIZE = 128


NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
TIM_TOKS = [f't{i}' for i in range(TIM_SIZE)]
VEL_TOKS = [f'v{i}' for i in range(VEL_SIZE)]

BOS_TOK = "BOS"
# Le token dummy sert seulement à initialiser les mots du vocab à partir de l'index 1, conformément aux prérequis de la fonction vocab()
VOCAB = ["dummy"] + [BOS_TOK] + NOTE_TOKS + DUR_TOKS + TIM_TOKS + VEL_TOKS 

DICT = [(element, index) for index, element in enumerate(VOCAB)]


custom_vocab = vocab(OrderedDict(DICT))
itos_vocab = custom_vocab.get_itos()
vocab_size = len(custom_vocab)


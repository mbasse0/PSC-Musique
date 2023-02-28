from torchtext.vocab import vocab
from collections import OrderedDict

#special tokens
SOS = "sos" #start
EOS = "eos" #end
#PAD = "xx"  #inutile si la data est de longueur définie

NOTE_SIZE = 128
DUR_SIZE = 160
TIM_SIZE = 1000
VEL_SIZE = 128

NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
TIM_TOKS = [f't{i}' for i in range(TIM_SIZE)]
VEL_TOKS = [f'v{i}' for i in range(VEL_SIZE)]

VOCAB_NOTE = [SOS,EOS] + NOTE_TOKS 
VOCAB_DUR = [SOS,EOS] + DUR_TOKS 
VOCAB_TIM = [SOS,EOS] + TIM_TOKS
VOCAB_VEL = [SOS,EOS] + VEL_TOKS

##créer 4 vocabs, possible ?
DICT = [(element, index) for index, element in enumerate(VOCAB)]

custom_vocab = vocab(OrderedDict(DICT))

def itos(voc, i):
    return voc.get_itos()[i]

def stoi(voc, s):
    return voc[s]

itos_vocab = custom_vocab.get_itos()

from torchtext.vocab import vocab
from collections import OrderedDict

#special tokens
SOS = "sos" #start

SPECIAL = [SOS]

NOTE_SIZE = 128
DUR_SIZE = 96 # double ronde max
TIM_SIZE = 192 # quadruple ronde max
VEL_SIZE = 128

NOTE_TOKS = [f'n{i}' for i in range(NOTE_SIZE)] 
DUR_TOKS = [f'd{i}' for i in range(DUR_SIZE)]
TIM_TOKS = [f't{i}' for i in range(TIM_SIZE)]
VEL_TOKS = [f'v{i}' for i in range(VEL_SIZE)]

# Le token dummy sert seulement à initialiser les mots du vocab à partir de l'index 1, conformément aux prérequis de la fonction vocab()
VOCAB_NOTE = ["dummy"] + NOTE_TOKS 
VOCAB_DUR = ["dummy"] + DUR_TOKS
VOCAB_TIM = ["dummy"] + TIM_TOKS
VOCAB_VEL = ["dummy"] + SPECIAL + VEL_TOKS

DICT_NOTE = [(element, index) for index, element in enumerate(VOCAB_NOTE)]
DICT_DUR = [(element, index) for index, element in enumerate(VOCAB_DUR)]
DICT_TIM = [(element, index) for index, element in enumerate(VOCAB_TIM)]
DICT_VEL = [(element, index) for index, element in enumerate(VOCAB_VEL)]

CV_NOTE = vocab(OrderedDict(DICT_NOTE))
CV_DUR = vocab(OrderedDict(DICT_DUR))
CV_TIM = vocab(OrderedDict(DICT_TIM))
CV_VEL = vocab(OrderedDict(DICT_VEL))

itos_NOTE = CV_NOTE.get_itos()
itos_DUR = CV_DUR.get_itos()
itos_TIM = CV_TIM.get_itos()
itos_VEL = CV_VEL.get_itos()

VOCAB_TOTAL = ["dummy"] + SPECIAL + NOTE_TOKS + DUR_TOKS + TIM_TOKS + VEL_TOKS
DICT_TOTAL = [(element, index) for index, element in enumerate(VOCAB_TOTAL)]
custom_vocab_4 = vocab(OrderedDict(DICT_TOTAL))
itos_vocab_4 = custom_vocab_4.get_itos()


def custom_vcb(tok):
    '''
    tok = [note,dur,tim,vel]
    '''
    return (CV_NOTE[tok[0]],
            CV_DUR[tok[1]],
            CV_TIM[tok[2]],
            CV_VEL[tok[3]])


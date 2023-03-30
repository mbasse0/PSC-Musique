import torch
from torch import Tensor
import torch.nn as nn
from JazzBot.vocab import *
import numpy as np

class tokenTypeLoss(nn.Module):
    """
    Custom loss function :
    uses CrossEntropyLoss and penalises errors on the type of token (pitch, duration, time and velocity)

    Inspired by https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1
    """

    def __init__(self, weight_=1.) -> None:
        super(tokenTypeLoss,self).__init__()
        self.weight = weight_

    def forward(self, output : Tensor, target : Tensor) -> Tensor:
        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        mask = Tensor([itos_vocab[output[i]][0] != itos_vocab[target[i]][0] for i in range(output.size)])
        high_cost = self.weight * (loss * mask.float()).mean()
        return loss + high_cost


class rhythmLoss(nn.Module):
    """
    Custom loss function :
    uses CrossEntropyLoss and penalises errors on the type of token (pitch, duration, time and velocity) and 
    weighs the penality on duration, time and velocity by considering the numerical distance between the output and the target

    Inspired by https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1
    """

    def __init__(self,coeff_) -> None:
        """
        coeff_ : float[5] :
            0 : weigh of token type error penalization
            1 : weigh of pitch penalization
            2 : weigh of duration penalization
            3 : weigh of time penalization
            4 : weigh of velocity penalization
        """
        super(rhythmLoss,self).__init__()
        self.coef = coeff_

    def forward(self, output : Tensor, target : Tensor) -> Tensor:
        target = torch.LongTensor(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        mask = torch.zeros_like(output)
        for i in range(output.size):
            if itos_vocab[output[i]][0] != itos_vocab[target[i]][0]:
                mask[i] = self.coef[0]
            else:
                if itos_vocab[output[i]] == 'p':
                    mask[i] = self.coef[1] * float(itos_vocab[output[i]] != itos_vocab[target[i]])
                elif itos_vocab[output[i]] == 'd':
                    mask[i] = self.coef[2] * np.abs((float(itos_vocab[output[i]][1:]) - float(itos_vocab[target[i]][1:])))/160
                elif itos_vocab[output[i]] == 't':
                    mask[i] = self.coef[3] * np.abs((float(itos_vocab[output[i]][1:]) - float(itos_vocab[target[i]][1:])))/100
                else:
                    mask[i] = self.coef[4] * np.abs((float(itos_vocab[output[i]][1:]) - float(itos_vocab[target[i]][1:])))/128

        high_cost = (loss * mask.float()).mean()
        return loss + high_cost

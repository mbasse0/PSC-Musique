import torch
from torch import Tensor
import torch.nn as nn
from vocab import *
import numpy as np
import torch.nn.functional as F

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # target = torch.LongTensor(target).to(device)
        criterion = nn.CrossEntropyLoss()
        # print("output", output, output.shape)
        # print("target", target, target.shape)

        softmax_output = F.softmax(output, dim=-1)

        max_indices = torch.argmax(softmax_output, dim=1)
        # print("soft output", softmax_output, softmax_output.shape)
        # print("max ind", max_indices, max_indices.shape)

        loss = criterion(output.to(device), target.to(device)).to(device)
        batch_size = len(max_indices)
        mask = torch.zeros((32, 120)).to(device)
        for i in range(batch_size):
            mask[i] = Tensor([itos_vocab[max_indices[i][j]][0] != itos_vocab[target[i][j]][0] for j in range(len(max_indices[0]))])
        # mask = Tensor([itos_vocab[output[i]][0] != itos_vocab[target[i]][0] for i in range(output.size)])
        high_cost = self.weight * (loss * mask.float()).mean()
        # return loss, pct_err
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

    def forward(self, output : Tensor, target : Tensor):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.CrossEntropyLoss()
        softmax_output = F.softmax(output, dim=-1)

        max_indices = torch.argmax(softmax_output, dim=1)

        batch_size, sequence_length = target.shape

        loss = criterion(output.to(device), target.to(device)).to(device)
        batch_size = len(max_indices)
        mask = torch.zeros((batch_size, sequence_length)).to(device)

        for i in range(batch_size):
            for j in range(sequence_length):
                if itos_vocab[max_indices[i][j]][0] != itos_vocab[target[i][j]][0]:
                    mask[i][j] = self.coef[0]
                else:
                    if itos_vocab[max_indices[i][j]][0] == 'p':
                        mask[i][j] = self.coef[1] * float(itos_vocab[max_indices[i][j]] != itos_vocab[target[i][j]][1:])
                    elif itos_vocab[max_indices[i][j]][0] == 'd':
                        mask[i][j] = self.coef[2] * np.abs((float(itos_vocab[max_indices[i][j]][1:]) - float(itos_vocab[target[i][j]][1:])))/160
                    elif itos_vocab[max_indices[i][j]][0] == 't':
                        mask[i][j] = self.coef[3] * np.abs((float(itos_vocab[max_indices[i][j]][1:]) - float(itos_vocab[target[i][j]][1:])))/100
                    else:
                        mask[i][j] = self.coef[4] * np.abs((float(itos_vocab[max_indices[i][j]][1:]) - float(itos_vocab[target[i][j]][1:])))/128

        high_cost = (loss * mask.float()).mean()
        pct_err = mask.float().mean().item()
        return loss + high_cost
    

class harmonicLoss(nn.Module):
    """
    Custom loss function :
    uses CrossEntropyLoss and penalises errors on the type of token (pitch, duration, time and velocity),
    weighs the penality on duration, time and velocity by considering the numerical distance between the output and the target,
    and weighs de penalty on pitch considering musical intervals (fifths, fourths...)

    Inspired by https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1
    """

    def __init__(self,coeff_,harmonic_) -> None:
        """
        coeff_ : float[4] :
            0 : weigh of token type error penalization
            1 : weigh of duration penalization
            2 : weigh of time penalization
            3 : weigh of velocity penalization
        harmonic_ : float[] :
            0 : weigh of the fifth
            1 : weigh of the fourth
            2 : weigh of the minor third
            3 : weigh of the major third
            4 : weigh of the minor second
            5 : weigh of the major second
            6 : weigh of other intervals

        For a first try:
            [0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 1.5]
        """
        super(harmonicLoss,self).__init__()
        self.coef = coeff_
        self.harmonic = harmonic_

    def forward(self, output : Tensor, target : Tensor):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.CrossEntropyLoss()
        softmax_output = F.softmax(output, dim=-1)

        batch_size, sequence_length = target.shape

        max_indices = torch.argmax(softmax_output, dim=1)

        loss = criterion(output.to(device), target.to(device)).to(device)
        batch_size = len(max_indices)
        mask = torch.zeros((batch_size, sequence_length)).to(device)

        for i in range(batch_size):
            for j in range(sequence_length):
                if itos_vocab[max_indices[i][j]][0] != itos_vocab[target[i][j]][0]:
                    mask[i][j] = self.coef[0]
                else:
                    if itos_vocab[max_indices[i][j]][0] == 'p':
                        mask[i][j] = self.harmonicWeigh(int(itos_vocab[max_indices[i][j]][1:]),int(itos_vocab[target[i][j]][1:]))
                    elif itos_vocab[max_indices[i][j]][0] == 'd':
                        mask[i][j] = self.coef[1] * np.abs((float(itos_vocab[max_indices[i][j]][1:]) - float(itos_vocab[target[i][j]][1:])))/160
                    elif itos_vocab[max_indices[i][j]][0] == 't':
                        mask[i][j] = self.coef[2] * np.abs((float(itos_vocab[max_indices[i][j]][1:]) - float(itos_vocab[target[i][j]][1:])))/100
                    else:
                        mask[i][j] = self.coef[3] * np.abs((float(itos_vocab[max_indices[i][j]][1:]) - float(itos_vocab[target[i][j]][1:])))/128

        high_cost = (loss * mask.float()).mean()
        pct_err = mask.float().mean().item()
        return loss + high_cost
    
    def harmonicWeigh(self,pitch1 : int, pitch2 : int):
        """
        Returns the penalty weigh by considering the "harmonic distance" betwen two pitches, using the self.harmonic coefficients
        Input : pitch1, pitch2 : int
                    The pitches considered
        Output : float
                    The weigh of the penalty
        """
        semi = np.abs(pitch1 - pitch2)
        if semi == 7:
            return self.harmonic[0]
        elif semi == 5:
            return self.harmonic[1]
        elif semi == 3:
            return self.harmonic[2]
        elif semi == 4:
            return self.harmonic[3]
        elif semi == 1:
            return self.harmonic[4]
        elif semi == 2:
            return self.harmonic[5]
        else:
            return self.harmonic[6]

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
        pct_err = mask.float().mean().item()
        # return loss, pct_err
        return (loss + high_cost), pct_err


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

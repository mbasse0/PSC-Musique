from torch import nn, optim
import math
from positional_encoding import *
#from config import *
from vocab import *
import pytorch_lightning as pl

class Transformer(pl.LightningModule):
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        learning_rate
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.lr = learning_rate

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first = True
        )
        self.out1 = nn.Linear(dim_model, num_tokens)
        self.out2 = nn.Linear(dim_model, num_tokens)
        self.out3 = nn.Linear(dim_model, num_tokens)
        self.out4 = nn.Linear(dim_model, num_tokens)

    # A modifier pour utiliser 4 out functions différentes selon les cas    
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        prev_token = tgt[:,-1]
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # Out size = (batch_size, sequence length, dim_model) 

        # Pour toutes les valeurs du batch size, on passe le résultat du transformer (de la taille de l'embeddding) dans la couche out adaptée afin d'obtenir un output final de la taille du vocab
        for d in range(len(prev_token)):
            type_tok = itos_vocab[prev_token[d]][0]
            if type_tok =='n':
                out = self.out1(transformer_out)
            elif type_tok=='d':
                out = self.out2(transformer_out)
            elif type_tok=='t':
                out = self.out3(transformer_out)
            elif type_tok=='v':
                out = self.out4(transformer_out)
        #outSize définie par la outSize de self.out1 (num_token)
        return out

    # Genere un masque triangulaire  
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    # Le pad mask sera utile quand on aura ajouté les PAD tokens
    # def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
    #     # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    #     # [False, False, False, True, True, True]
    #     return (matrix == pad_token)

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch):
        y_input, y_expected = batch
        #y_input, y_expected sont tokenizés ne sont pas embeddés ce sont des batchs de séquences de nombres entre 0 et 1417
        # appartiennent à [0,1417]^(120)^(batchsize)

        # X est ce qu'on donne à l'encoder. Un vecteur nul dans notre cas en l'absence d'informations contextuelles
        X = torch.tensor([0]*len(y_input))
        #y_input, y_expected = y_input.to(self.device), y_expected.to(self.device)
        X, y_input, y_expected = X.to(self.device) , y_input.to(self.device), y_expected.to(self.device)

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)

        # Standard training except we pass in y_input and tgt_mask
        pred = self(X, y_input, tgt_mask)
        #pred est embédé, chaque token est un vetceur one hot de {0,1}^1417
        #donc pred appartient à {0,1}^1417^120^batchsize

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)
        lossF = nn.CrossEntropyLoss()
        loss = lossF(pred, y_expected)
        self.log('Training loss', loss)
        return loss
    
    def validation_step(self, batch, batchidx):
        y_input, y_expected = batch
        #y_input, y_expected sont tokenizés ne sont pas embeddés ce sont des batchs de séquences de nombres entre 0 et 1417
        # appartiennent à [0,1417]^(120)^(batchsize)

        # X est ce qu'on donne à l'encoder. Un vecteur nul dans notre cas en l'absence d'informations contextuelles
        X = torch.tensor([0]*len(y_input))
        #y_input, y_expected = y_input.to(self.device), y_expected.to(self.device)
        X, y_input, y_expected = X.to(self.device).clone().detach() , y_input.to(self.device).clone().detach(), y_expected.to(self.device).clone().detach()

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = self.get_tgt_mask(sequence_length).to(self.device)

        # Standard training except we pass in y_input and tgt_mask
        pred = self(X, y_input, tgt_mask)
        #pred est embédé, chaque token est un vetceur one hot de {0,1}^1417
        #donc pred appartient à {0,1}^1417^120^batchsize

        # Permute pred to have batch size first again
        pred = pred.permute(0, 2, 1)
        lossF = nn.CrossEntropyLoss()
        loss = lossF(pred, y_expected)
        self.log('Validation loss', loss)
        return loss
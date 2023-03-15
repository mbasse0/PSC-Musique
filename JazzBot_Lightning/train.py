from model import *
from data_loader import *
from config import *
import pytorch_lightning as pl

model = Transformer(
   num_tokens=len(CV), dim_model=256, num_heads=2, num_encoder_layers=1, num_decoder_layers=6, dropout_p=0.1
)

opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

dataloader = dataloader("../Data/", batch_size=32, N=25)

trainer = pl.trainer()
trainer.fit(model, dataloader, dataloader)

torch.save(model, "../model.pth")
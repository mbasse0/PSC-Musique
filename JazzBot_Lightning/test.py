from train import *
from config import *


model = Transformer(
    num_tokens=len(CV), dim_model=256, num_heads=2, num_encoder_layers=1, num_decoder_layers=6, dropout_p=0.1
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
dataloader = dataloader(path,batch_size=32,N=N)

train_loss_list = fit(model, opt, loss_fn, dataloader, 10)


toks = midiToTokens(path, "A Thousand Miles - Vanessa Carlton - Verse-And-Pre-Chorus.mid")
y_input = torch.tensor(pieceToInputTarget(tokensToPieces(toks,N)[0])[0]).to(device)
tgt_mask = model.get_tgt_mask(y_input[0].size(0)).to(device)
pred = model(torch.tensor([0]*len(N+4)).to(device), y_input, tgt_mask)
next_item = pred.topk(1)[1].view(-1)[-1].item()
print(itos[next_item])
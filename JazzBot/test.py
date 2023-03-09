from config import *
from model import *
from data_processor import *

# model = torch.load("../model.pth")
# model.eval()


toks = midiToTokens(path, "A Thousand Miles - Vanessa Carlton - Verse-And-Pre-Chorus.mid")
y_input = torch.tensor(pieceToInputTarget(tokensToPieces(toks,N)[0])[0]).to(device)
print(pieceToInputTarget(tokensToPieces(toks,N)[0]))
# tgt_mask = model.get_tgt_mask(y_input.size(0)).to(device)
# pred = model(torch.tensor([0]*(N+3)).to(device), y_input, tgt_mask)
# next_item = pred.topk(1)[1].view(-1).item()
# print([pred.topk(1)[1].view(-1)[i].item() for i in range(100)])
# print(itos[next_item])
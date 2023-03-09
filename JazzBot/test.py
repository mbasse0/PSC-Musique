from config import *
from model import *
from data_processor import *

model = torch.load("../model.pth")
model.eval()


toks = midiToTokens(path, "A Thousand Miles - Vanessa Carlton - Verse-And-Pre-Chorus.mid")
y_input = torch.tensor(pieceToInputTarget(tokensToPieces(toks,4*N)[0])[0]).to(device)
print(len(y_input))
tgt_mask = model.get_tgt_mask(y_input.size(0)).to(device)
pred = model(torch.tensor([0]*(4*N-1)).to(device), y_input, tgt_mask)
# Get the index of the highest probability for each token in the sequence
predicted_tokens = torch.argmax(pred, dim=2)

# Convert the indices to token strings using the CV vocabulary
input_strings = [itos[token.item()] for token in y_input]
predicted_strings = [itos[token.item()] for token in predicted_tokens[0]]

# Print the predicted token sequence as a string
print(" ".join(predicted_strings))
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import music21 as m21
import os

from config import *
from model import *
from data_processor import *
from vocab import *
from decoding import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load("./modelperf.pth",map_location=torch.device(device))
model = Transformer(num_tokens=len(CV), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1).to(device)
model.load_state_dict(state_dict)
model.eval()

file = st.file_uploader("Upload a MIDI file",type=['mid'])
with open(os.path.join("./midi_data/",file.name),"wb") as f:
    f.write(file.getbuffer())

if file is not None:
    clicked = st.button('Generate')

if clicked:
    toks = midiToTokens("./midi_data/", file.name)
    y_input = torch.tensor(pieceToInputTarget(tokensToPieces(toks,4*N)[0])[0]).to(device)
    tgt_mask = model.get_tgt_mask(y_input.size(0)).to(device)
    pred = model(torch.tensor([0]*(4*N-1)).to(device), y_input, tgt_mask)
    # Get the index of the highest probability for each token in the sequence
    predicted_tokens = torch.argmax(pred, dim=2)

    # Convert the indices to token strings using the CV vocabulary
    input_strings = [itos[token.item()] for token in y_input]
    predicted_strings = [itos[token.item()] for token in predicted_tokens[0]]

    # Print the predicted token sequence as a string
    print(" ".join(predicted_strings))

    tokens_to_midi(predicted_tokens,"result.mid")

    st.download_button("Download","result.mid")
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import music21 as m21
import os

from config import *
from model import *
from vocab import *
from data_decoder import *
from data_encoder import *
from generate import *

st.title("JazzBot")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load("./model4out.pth",map_location=torch.device(device))
model = Transformer(num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1).to(device)
model.load_state_dict(state_dict)
model.eval()

file = st.file_uploader("Upload a MIDI file",type=['mid'])

bpm = st.slider("BPM",min_value=60,max_value=200,value=120)
maxl = st.slider("Maximum length of the generated sequence",min_value=20,max_value=400,value=100)

if file is not None:
    with open(os.path.join("./app_data/", file.name), "wb") as f:
        f.write(file.getbuffer())

clicked = False
if file is not None:
    clicked = st.button('Generate')



if clicked:
    start_tokens = midiToTokens("./app_data/"+ file.name)
    print("taille initiale", len(start_tokens))
    print("taille_generée:", maxl)
    print("Device:", device)

    start_tokens = [custom_vocab[el] for el in start_tokens]
    # Generate a sequence of tokens
    generated_tokens = generate_sequence(model, start_tokens, max_length=maxl+len(start_tokens), temperature=1.0)
    
    #On peut sauver la prédiction dans un array
    #np.save("generation4.npy", decoded_tokens)

    ## CONVERSION DE LA SEQUENCE EN MIDI
    generated_tokens = [itos_vocab[el] for el in generated_tokens]
    tokens_to_midi(generated_tokens, "result.mid", bpm)
    # st.download_button("Download", data="result.mid", file_name="result.mid", mime="audio/midi")

    with open("result.mid", "rb") as f:
        midi_data = f.read()

    st.download_button("Download", data=midi_data, file_name="result.mid", mime="audio/midi")


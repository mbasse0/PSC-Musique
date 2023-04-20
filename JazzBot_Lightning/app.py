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
st.text("Use JazzBot to create a jazz solo")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load("./model4out.pth",map_location=torch.device(device))
model = Transformer(num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1).to(device)
model.load_state_dict(state_dict)
model.eval()

file = st.file_uploader("Upload a MIDI file",type=['mid'])

temp = st.slider("Temperature",min_value=0.5,max_value=1.5,value=1.0)
bpm = st.slider("BPM",min_value=60,max_value=200,value=120)
maxl = st.slider("Maximum length of the generated sequence",min_value=20,max_value=400,value=100)
generated = False


if file is not None:
    with open(os.path.join("./app_data/", file.name), "wb") as f:
        f.write(file.getbuffer())
    st.session_state.midi_data = file.getbuffer().tobytes()  # Add this line to store the MIDI data in the session state



#Fonction qui permet de charger le html du piano roll dans l'app streamlit
def html_loader(midi_data=None, load_generated=False):
    with open("webaudio_pianoroll.html") as f:
        html = f.read()
        
    if load_generated:
        js_code = f"""
        <script>
            document.addEventListener("DOMContentLoaded", function(event) {{
                window.loadGeneratedMidiData(new Uint8Array({list(midi_data)}));
            }});
        </script>
        """
        html += js_code

    elif midi_data:
        js_code = f"""
        <script>
            document.addEventListener("DOMContentLoaded", function(event) {{
                window.loadMidiData(new Uint8Array({list(midi_data)}));
            }});
        </script>
        """
        html += js_code

    
    components.html(html, height=600, width=1500, scrolling=False)


progress_bar = st.progress(0)

def progress_callback(progress):
    print(progress)
    progress = min(progress, 1.0)
    progress_bar.progress(progress)


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
    generated_tokens = generate_sequence(model, start_tokens, max_length=maxl+len(start_tokens), temperature=temp, progress_callback=progress_callback)
    
    #On peut sauver la prédiction dans un array
    #np.save("generation4.npy", decoded_tokens)

    ## CONVERSION DE LA SEQUENCE EN MIDI
    generated_tokens = [itos_vocab[el] for el in generated_tokens]
    tokens_to_midi(generated_tokens, "result.mid", bpm)

    with open("result.mid", "rb") as f:
        midi_data = f.read()
        gen_midi_data = midi_data
    st.download_button("Download", data=midi_data, file_name="result.mid", mime="audio/midi")
    generated = True
    html_loader(gen_midi_data, generated)

# webaudio_pianoroll()
# Modify the webaudio_pianoroll() function call like this:

if not generated:
    html_loader(midi_data = st.session_state.get("midi_data", None))

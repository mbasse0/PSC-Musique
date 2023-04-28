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
import music21

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide"
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
)


st.title("JazzBot")
st.text("Générer un solo de Jazz à partir d'un fichier MIDI")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load("./Models/model_10_deter_defaultloss_512_8_1_4_0.1_0.05.pth",map_location=torch.device(device))
#model = Transformer(num_tokens=len(custom_vocab), dim_model=512, num_heads=8, num_encoder_layers=1, num_decoder_layers=4, dropout_p=0.1, learning_rate=0.05).to(device)
#model.load_state_dict(state_dict)
#model.eval()



file = st.file_uploader("Charger un fichier MIDI",type=['mid'])

temp = st.slider("Température",min_value=0.5,max_value=1.5,value=1.0)
bpm = 120
maxl = st.slider("Nombre de notes à générer",min_value=1,max_value=200,value=25)
maxl *=4

generated = False


if file is not None:
    with open(os.path.join("./app_data/", file.name), "wb") as f:
        f.write(file.getbuffer())
    st.session_state.midi_data = file.getbuffer().tobytes()  # Add this line to store the MIDI data in the session state


if file is not None:
    midi_file = m21.midi.MidiFile()
    midi_file.open(os.path.join("./app_data/", file.name))
    midi_file.read()
    midi_file.close()
    str = midi.translate.midiFileToStream(midi_file)
    for el in str.recurse():
        if isinstance(el, tempo.MetronomeMark):
            bpm = el
            break


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
    progress = min(progress, 1.0)
    progress_bar.progress(progress)


clicked = False
if file is not None:
    clicked = st.button('Générer')



if clicked:
    start_tokens = midiToTokens("./app_data/"+ file.name)
    print("taille initiale", len(start_tokens))
    print("taille_generée:", maxl)
    print("Device:", device)

    print("les start tokens", start_tokens)
    start_tokens = [custom_vocab[el] for el in start_tokens]
    # Generate a sequence of tokens
    generated_tokens = generate_sequence(model, start_tokens, max_length=maxl+len(start_tokens), temperature=temp, progress_callback=progress_callback)
    
    #On peut sauver la prédiction dans un array
    #np.save("generation4.npy", decoded_tokens)

    ## CONVERSION DE LA SEQUENCE EN MIDI
    generated_tokens = [itos_vocab[el] for el in generated_tokens]
    tokens_to_midi(generated_tokens, "./Results/result.mid", bpm)

    with open("./Results/result.mid", "rb") as f:
        midi_data = f.read()
        gen_midi_data = midi_data
    st.download_button("Download", data=midi_data, file_name="result.mid", mime="audio/midi")
    generated = True
    html_loader(gen_midi_data, generated)



if not generated:
    html_loader(midi_data = st.session_state.get("midi_data", None))

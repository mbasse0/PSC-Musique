import numpy as np
from vocab import *
from music21 import *
import os



def noteToToken(n,l):
    '''
    parameters : note, last_time
    NB : A terme il faudra arrondir plutot que de prendre la partie entiere
    '''
    note_pitch = n.pitch.midi
    last_time = l
    note_duration = int(n.duration.quarterLength*4)
    note_offset = int(n.offset*4 - last_time)
    last_time = n.offset*4
    note_velocity = n.volume.velocity

    return les_tokens.append([NOTE_TOKS[note_pitch],
                            DUR_TOKS[note_duration],
                            TIM_TOKS[note_offset],
                            VEL_TOKS[note_velocity]])

def midiToTokens(midi_file):
    '''
    read a midi file and extract the notes, no token start or end
    '''
    tokens = []
    # Create a stream from the MIDI file
    stream = midi.translate.midiFileToStream(midi_file)
    # Iterate over the notes in the stream and extract the note information
    last_time = 0
    for note in stream.flat.notes:
        if note.isNote:
            tokens.append(noteToToken(note,last_time))

        if note.isChord:
            for n in note:
                tokens.append(noteToToken(n,last_time))

    return tokens

def tokensToPieces(t,N):
    '''
    cut the list of tokens into pieces of length
    '''
    pieces = []
    for i in range(len(t)//(N+1)-1):
        pieces.append(t[i:i+taille_bloc])
    return pieces

def pieceToInputTarget(p):
    '''
    return a couple of vectors (input, target) with input = SOS + piece; target = piece + EOS
    '''
    target = vectorize()



folder_path = "../Data" 
    
# Load the MIDI file
midi_file = midi.MidiFile()

les_tokens = []

# Get all the file names in the folder
file_names = os.listdir(folder_path)
for f in file_names:
    print(f)
    midi_file = midi.MidiFile()
    midi_file.open(folder_path +f)
    midi_file.read()
    midi_file.close()
    # Create a stream from the MIDI file
    stream = midi.translate.midiFileToStream(midi_file)

    # Iterate over the notes in the stream and extract the note information
    last_time = 0

    for note in stream.flat.notes:
        if note.isNote:
            les_tokens.append(noteToToken(note))

        if note.isChord:
            for n in note:
                les_tokens.append(noteToToken(n))


#répartir le data du morceau en blocs de 120 attributs (30 notes)
#Et associer à chaque bloc la réponse attendue (l'attribut suivant)

taille_bloc = 120
les_morceaux = []
les_morceaux_shift = []

for i in range(len(les_tokens)//(taille_bloc+1)-1):
    les_morceaux.append(les_tokens[i:i+taille_bloc])
    les_morceaux_shift.append(les_tokens[i+1:i+1+taille_bloc])


def vectorize(i):
    return [0]*i + [1] + [0]* (len(custom_vocab)-i-1)
def unvectorize(v):
    for i in range(len(v)):
        if v[i]:
            return i

input_vect = [ [vectorize(custom_vocab[tok]) for tok in morceau] for morceau in les_morceaux ]
rep_vect = [ [vectorize(custom_vocab[tok]) for tok in morceau] for morceau in les_morceaux_shift ]
        

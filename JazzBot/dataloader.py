import numpy as np
from vocab import *
from music21 import *
import os

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
            note_pitch = note.pitch.midi
            # A terme il faudra arrondir plutot que de prendre la partie entiere
            note_duration = int(note.duration.quarterLength*4)
            note_offset = int(note.offset*4 - last_time)
            last_time = note.offset*4
            note_velocity = note.volume.velocity
            les_tokens.append(NOTE_TOKS[note_pitch])
            les_tokens.append(DUR_TOKS[note_duration])
            les_tokens.append(TIM_TOKS[note_offset])
            les_tokens.append(VEL_TOKS[note_velocity])
            # print("Note Pitch:", note_pitch)
            # print("Note Duration:", note_duration)
            # print("Note TimeShift:", note_offset)
            # print("Note Velocity:", note_velocity)

        if note.isChord:

            for note2 in note:
                note_pitch = note2.pitch.midi
                note_duration = int(note.duration.quarterLength*4)
                note_offset = int(note.offset*4 - last_time)
                last_time = note.offset*4
                note_velocity = note2.volume.velocity
                les_tokens.append(NOTE_TOKS[note_pitch])
                les_tokens.append(DUR_TOKS[note_duration])
                les_tokens.append(TIM_TOKS[note_offset])
                les_tokens.append(VEL_TOKS[note_velocity])
                # print("Note Pitch:", note_pitch)
                # print("Note Duration:", note_duration)
                # print("Note Time:", note_offset)
                # print("Note Velocity:", note_velocity)


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
          

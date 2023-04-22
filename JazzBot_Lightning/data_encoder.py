from vocab import *
from tqdm import tqdm
from music21 import *
import os

def midiToTokens(filename):


    # Load the MIDI file
    midi_file = midi.MidiFile()

    les_tokens = []
    midi_file.open(filename)
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
            note_duration = round(note.duration.quarterLength*12)
            note_offset = round(note.offset*12 - last_time)
            last_time = note.offset*12
            note_velocity = note.volume.velocity
            les_tokens.append(NOTE_TOKS[note_pitch])
            les_tokens.append(DUR_TOKS[note_duration])
            if note_offset < NOTE_SIZE:
                les_tokens.append(TIM_TOKS[note_offset])
            les_tokens.append(VEL_TOKS[note_velocity])

        if note.isChord:

            for note2 in note:
                note_pitch = note2.pitch.midi
                note_duration = int(note.duration.quarterLength*4)
                note_offset = int(note.offset*12 - last_time)
                last_time = note.offset*12
                note_velocity = note2.volume.velocity
                les_tokens.append(NOTE_TOKS[note_pitch])
                les_tokens.append(DUR_TOKS[note_duration])
                if note_offset < NOTE_SIZE:
                    les_tokens.append(TIM_TOKS[note_offset])
                les_tokens.append(VEL_TOKS[note_velocity])
    return les_tokens



def midifolderToTokens(folder_path):
    les_tokens = []

    # Get all the file names in the folder
    file_names = os.listdir(folder_path)
    for f in tqdm(file_names):
        print(f)
        midi_file = midi.MidiFile()
        midi_file.open(folder_path + "/" +f)
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
                note_duration = int(note.duration.quarterLength*12)
                note_offset = int(note.offset*12 - last_time)
                last_time = note.offset*12
                note_velocity = note.volume.velocity
                les_tokens.append(NOTE_TOKS[note_pitch])
                les_tokens.append(DUR_TOKS[note_duration])
                if note_offset < NOTE_SIZE:
                    les_tokens.append(TIM_TOKS[note_offset])
                les_tokens.append(VEL_TOKS[note_velocity])

            if note.isChord:

                for note2 in note:
                    note_pitch = note2.pitch.midi
                    note_duration = int(note.duration.quarterLength*12)
                    note_offset = int(note.offset*12 - last_time)
                    last_time = note.offset*12
                    note_velocity = note2.volume.velocity
                    les_tokens.append(NOTE_TOKS[note_pitch])
                    les_tokens.append(DUR_TOKS[note_duration])
                    if note_offset < NOTE_SIZE:
                        les_tokens.append(TIM_TOKS[note_offset])
                    les_tokens.append(VEL_TOKS[note_velocity])
    return les_tokens

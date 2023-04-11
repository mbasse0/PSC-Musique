import music21
from music21 import * 
from vocab import *

def tokens_to_stream(token_array):
    # token_array attend un array de tokens de type string (par exemple ['n58','d2','t5','v106']])
    # Initialize variables to keep track of note properties
    current_pitch = None
    current_duration = None
    current_offset = 0.0
    current_velocity = None
    
    # Create an empty stream
    stream = music21.stream.Stream()

    # Iterate through the token array and create notes
    for token in token_array:
        if token.startswith("n"):
            # Token represents a pitch
            current_pitch = int(token[1:])
        elif token.startswith("d"):
            # Token represents a duration
            current_duration = int(token[1:])
        elif token.startswith("t"):
            # Token represents a time shift
            current_offset += int(token[1:])/4
        elif token.startswith("v"):
            # Token represents a velocity
            current_velocity = int(token[1:])
        else:
            # Token is not recognized, skip
            continue

        # If all note properties have been set, create a note and add it to the stream
        if current_pitch is not None and current_duration is not None and current_velocity is not None:
            # Create a note object with the current properties
            note = music21.note.Note()
            pitch = music21.pitch.Pitch()
            pitch.midi = current_pitch
            note.pitch = pitch
            note.duration = music21.duration.Duration(current_duration/4)
            note.volume.velocity = current_velocity
            note.offset = current_offset

            # Add the note to the stream
            stream.append(note)

            # Reset note properties
            current_pitch = None
            current_duration = None
            current_velocity = None

    return stream

# Define a function to convert an array of tokens to a MIDI file
def tokens_to_midi(token_array, filename, BPM):
    # Convert the token array to a music21 Stream
    stream = tokens_to_stream(token_array)

    # Set the BPM
    mm = music21.tempo.MetronomeMark(number=BPM)
    stream.insert(0, mm)

    # Create a MIDI file from the Stream and save it
    midi_file = music21.midi.translate.streamToMidiFile(stream)
    midi_file.open(filename, "wb")
    midi_file.write()
    midi_file.close()
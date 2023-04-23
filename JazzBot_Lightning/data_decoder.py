import music21
from music21 import * 

def tokens_to_stream(token_array):
    '''
    turn a [n,d,t,v,...] array into a music21.stream
    '''
    
    # Initialize variables to keep track of note properties
    current_pitch = None
    current_duration = None
    current_offset = 0.0
    current_velocity = None
    previous_duration = None

    # Create an empty stream
    stream = music21.stream.Stream()

    # Iterate through the token array and create notes
    
    for token in token_array:
        if token.startswith("n"):
            # Token represents a pitch
            current_pitch = int(token[1:])
        elif token.startswith("d"):
            # Token represents a duration
            current_duration = int(token[1:])/12
        elif token.startswith("t"):
            # Token represents a time shift
            time_shift_duration = int(token[1:])/12
            current_offset += time_shift_duration #offset jamais réactualisé
            if previous_duration is not None:
                time_rest = time_shift_duration -  previous_duration 
                # Create a rest for the time shift duration and add it to the stream
                if time_rest>0:
                    rest = music21.note.Rest()
                    rest.duration.quarterLength = time_rest
                    rest.offset = current_offset - time_shift_duration
                    stream.append(rest)
        
        elif token.startswith("v"):
            # Token represents a velocity
            current_velocity = int(token[1:])
        else:
            # Token is not recognized, skip
            continue

        # If all note properties have been set, create a note and add it to the stream
        if current_pitch is not None and current_velocity is not None and current_duration is not None:
            # Create a note object with the current properties
            note = music21.note.Note()
            pitch = music21.pitch.Pitch()
            pitch.midi = current_pitch
            note.pitch = pitch
            note.duration = music21.duration.Duration(current_duration)
            note.volume.velocity = current_velocity
            note.offset = current_offset

            # Add the note to the stream
            stream.append(note)

            # Reset note properties
            current_pitch = None
            previous_duration = current_duration
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

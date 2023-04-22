from vocab import *
from tqdm import tqdm
from music21 import *
import os

#############################################################################################################################################

def noteToToken(n,last_time):
    '''
    # parameters : note, last_time
    '''
    note_pitch = n.pitch.midi
    note_duration = round(n.duration.quarterLength*12)
    note_timeshift = round(n.offset*12 - last_time)
    note_velocity = n.volume.velocity
    return [NOTE_TOKS[note_pitch],
            DUR_TOKS[note_duration],
            TIM_TOKS[note_timeshift],
            VEL_TOKS[note_velocity]]

def chordToToken(chord, last_time):
    tok = []
    note_offset = round(n.offset*12 - last_time)
    for n in chord:
        note_pitch = n.pitch.midi
        note_duration = round(n.duration.quarterLength*12)
        note_velocity = n.volume.velocity
        tok+=[NOTE_TOKS[note_pitch],
                DUR_TOKS[note_duration],
                TIM_TOKS[note_offset],
                VEL_TOKS[note_velocity]]
    return tok


#############################################################################################################################################
def midiToTokens(filename):
    '''
    parameter : folder_path,f = name of the midi file
    read a midi file and extract the notes in several arrays of tokens (cut where issues occur), return only the first array
    '''
    tokens = []
    midi_file = midi.MidiFile()
    midi_file.open(filename)
    midi_file.read()
    midi_file.close()
    stream = midi.translate.midiFileToStream(midi_file)
    last_time = 0
    tokens = []
    tok = []
    start = True
    for note in  stream.recurse().notes:
        if start and note.isRest:
            last_time = note.duration.quarterLength*12
            start = False
        if note.isNote:
            time_rest = note.offset*12-last_time
            # print(note, time_rest)
            if  time_rest < 0 or time_rest >= 192 or note.duration.quarterLength*12>=96:
                if(len(tok)>=120):
                    tokens.append(tok)
                    tok = []
            else :
                if note.isNote:
                    tok+=noteToToken(note,last_time)

                if note.isChord:
                    tok+=chordToToken(note,last_time)
            last_time = note.offset*12   
    if (len(tok)>=120):
        tokens.append(tok)

    return tokens[0]

##########################################################################################################################################################

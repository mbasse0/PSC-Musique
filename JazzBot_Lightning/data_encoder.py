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
    read a midi file and return the first starting_tokens (until exception raised #1)
    '''
    tokens = []
    midi_file = midi.MidiFile()
    midi_file.open(filename)
    midi_file.read()
    midi_file.close()
    stream = midi.translate.midiFileToStream(midi_file)
    tokens = []
    tok = []
    start = True
    for note in stream.flat.notes:
        if start :
            last_time = note.offset*12
            start = False
        if note.isNote:
            time_rest = note.offset*12-last_time
            if  time_rest >= 192 or note.duration.quarterLength*12>=96: #exception #1
                if(len(tok)>=121): 
                    tokens.append(tok)
                    tok = []
            else :
                if note.isNote:
                    tok+=noteToToken(note,last_time)

                if note.isChord:
                    print("accord")
                    tok+=chordToToken(note,last_time)
            last_time = note.offset*12   
    if (len(tok)>=1): #######################################changer avant process
        tokens.append(tok)
    return tokens[0]

##########################################################################################################################################################

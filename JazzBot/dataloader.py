import numpy as np
from vocab import *
from music21 import *
from torch.utils.data import Dataset, DataLoader
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

    return [NOTE_TOKS[note_pitch],
            DUR_TOKS[note_duration],
            TIM_TOKS[note_offset],
            VEL_TOKS[note_velocity]]

def midiToTokens(folder_path,f):
    '''
    parameter : folder_path,f = name of the midi file
    read a midi file and extract the notes, no token start or end
    '''
    tokens = []
    midi_file = midi.MidiFile()
    midi_file.open(folder_path +f)
    midi_file.read()
    midi_file.close()
    stream = midi.translate.midiFileToStream(midi_file)
    # Iterate over the notes in the stream and extract the note information
    last_time = 0
    for note in stream.flat.notes:
        if note.isNote:
            tokens.append(noteToToken(note,last_time))

        if note.isChord:
            for n in note:
                tokens.append(noteToToken(n,last_time))
    tokens.append([EOS]*4)

    return tokens

def tokensToPieces(t,N):
    '''
    cut the list of tokens into pieces of length N, add PAD if necessary
    '''
    pieces = []
    for i in range(len(t)//(N+1)-1):
        pieces.append(t[i:i+N])
        if i == len(t)//(N+1)-2:
            n = t[i:].length()
            list_pad = [PAD]*4
            pieces.append(t[i:]+(N-n)*[list_pad])
    return pieces

def pieceToInputTarget(p):
    '''
    return a couple of vectors (input, target) with input = SOS + piece; target = piece + EOS
    '''
    input_p = [[SOS]*4]+[custom_vocab(tok) for tok in p]
    target_p = [custom_vocab(tok) for tok in p]+[[EOS]*4]
    return(input_p,target_p)

def midisToVectInputTarget(folder_path,N):
    '''
    parameters : folder_path, N = size of pieces
    '''
    vectInput = []
    vectTarget = []
    file_names = os.listdir(folder_path)
    for f in file_names:
        tokens = midiToTokens(folder_path,f)
        pieces = tokensToPieces(t,N)
        for p in pieces:
            input,target = pieceToInputTarget(p)
            vectInput.append(input)
            vectTarget.append(target)
    return vectInput,vectTarget
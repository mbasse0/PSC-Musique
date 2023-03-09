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
        last_time = note.offset*4
        if note.isNote:
            tokens+=noteToToken(note,last_time)

        if note.isChord:
            for n in note:
                tokens+=noteToToken(n,last_time)

    return tokens

def tokensToPieces(t,n):
    '''
    parameters : t= tokens (4*(nb_tok))  
    cut the list of tokens into pieces of length n, add PAD if necessary
    '''
    pieces = []
    nb_tok = (len(t))//4
    for i in range(nb_tok-n+1):
        pieces.append(t[4*i:4*i+n])
    return pieces

def pieceToInputTarget(p):
    '''
    return a couple of vectors (input, target) with input = SOS + piece - last_note; target = piece 
    '''
    input_p = cv_sos()
    target_p = []
    for i in range(len(p)//4-1):
        input_p+= custom_vocab(p[4*i:4*(i+1)])
        target_p += custom_vocab(p[4*i:4*(i+1)])
    target_p += custom_vocab(p[4*(len(p)//4-1):4*(len(p)//4)])
    return(input_p,target_p)

def folderToVectInputTarget(folder_path,N):
    '''
    parameters : folder_path, N = number of notes in pieces 
    '''
    vectInput = []
    vectTarget = []
    file_names = os.listdir(folder_path)
    for f in file_names:
        print(f)
        tokens = midiToTokens(folder_path,f)
        pieces = tokensToPieces(tokens,4*N)
        for p in pieces:
            input,target = pieceToInputTarget(p)
            vectInput.append(input)
            vectTarget.append(target)
    return vectInput,vectTarget
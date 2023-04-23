from vocab import *
from music21 import *
from tqdm import tqdm
import csv


def tokensToPieces(t,N):
    '''
    parameters : t = tokens, 1 array
    cut the list of tokens into pieces of length N
    '''
    pieces = []
    for i in range(len(t)-N-1):
        pieces.append(t[i:i+N+1])
    return pieces

def pieceToInputTarget(p):
    '''
    return a couple of vectors (input, target) with input = SOS + piece - last_note; target = piece 
    '''
    CV_p = [custom_vocab[x] for x in p]
    input_p = CV_p[:-1]
    target_p = CV_p[1:]
    return input_p,target_p



def tokensFileToVectInputTarget(nameFile,N):
    '''
    parameters : folder_path, N = number of notes in pieces 
    read the file
    '''
    vectInput = []
    vectTarget = []
    with open(nameFile, mode = 'r') as file:
        reader = csv.reader(file)
        for tok_row in tqdm(reader):
            pieces = tokensToPieces(tok_row,N)
            for p in pieces:
                input,target = pieceToInputTarget(p)
                vectInput.append(input)
                vectTarget.append(target)
    return vectInput,vectTarget
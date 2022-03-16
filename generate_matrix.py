import os
import os.path
import ast
import sys
import numpy as np
from pathlib import Path
import math

ALPHABET = "ARNDCQEGHILKMFPSTWYV"

args = sys.argv

def softmax(z):
    """Compute softmax values for each sets of scores in x."""
    #assert len(z.shape) == 2
    e_x = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def Search(path, length = None):
    files = []

    for f in os.listdir(path):
        file = path + "/" + f
        if os.path.isdir(file):
            if length is None or str(length) in file:
                files.extend(Search(file))
        if file.endswith(".csv"):
            name = Path(file).stem
            if len(args) == 0 or name == "mutation_log":
                files.append(file)
                print(file)

    return files

def Analyze(path, files):
    mutations = []

    for file in files:
        with open(file, newline='') as r:
            lines = r.readlines()
            line_n = 0
            for line in lines:
                try:
                    mut = ast.literal_eval(line)
                    mut.append(line_n) #add approx step number
                    mutations.append(mut) #save mutation in list
                    line_n += 1
                except: print(line)


    print(len(mutations))

    dict = {}

    for i in range(len(ALPHABET)):
        aa = ALPHABET[i]
        dict[aa] = i

    print(dict)

    matrix = [ [0] * len(ALPHABET) for i in range(len(ALPHABET))]

    vector = [ [0] * len(ALPHABET) for i in range(290)]

    for i in range(len(vector)):
        for j in range(len(vector[i])):
            vector[i][j] = [0,0]

    for i in range(len(ALPHABET)):
        for j in range(len(ALPHABET)):
            matrix[i][j] = [0,1]

    for mut in mutations:
        pos = mut[0]
        aa1 = dict[mut[1]]
        aa2 = dict[mut[2]]
        score = mut[4]
        sub = mut[3]
        accepted = mut[5]
        step = mut[6]

        if aa1 == aa2: continue #no change (???)
        #if not accepted: continue #accepted
        if score > 0.2: continue #mutation score

        vector[pos][aa2][0] += math.exp(-score)-1
        vector[pos][aa1][0] += .333*(math.exp(score)-1)

        vector[pos][aa2][1] += 1
        vector[pos][aa1][1] += .333

        matrix[aa1][aa2][0] += (math.exp(-score)-1)*step**.5
        matrix[aa1][aa2][1] += 1

        #if pos == 289: print(mut[1],mut[2],score)



    for line in matrix:
        l = ""
        for v in line:
            if v[1] == 1: v[0] = -10
            l += f"{v[0]/v[1]:4.2f} "
        print(l)

    print()

    for line in matrix:
        l = ""
        for v in line:
            l += f"{v[1]:4.2f} "
        print(l)

    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j][1] > 0: vector[i][j][0] = vector[i][j][0]/vector[i][j][1]

    minimum_score = np.min(vector)

    print(minimum_score)

    scores = [ [0] * len(ALPHABET) for i in range(290)]

    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j][1] == 0: scores[i][j] = 2 * minimum_score
            else: scores[i][j] = vector[i][j][0]

    scores = softmax(scores)

    for pos in scores:
        l = ""
        for aa in pos:
            l += f"{aa} "
        #print(l)

    seq = ""
    for pos in scores:
        idx = np.argmax(pos,axis=0)
        seq += ALPHABET[idx]

    print()
    print(np.min(scores),np.max(scores))
    print()
    print(seq)

def Main():
    L = 290
    path = "."

    if len(args) > 1:
        print("PATH: " + args[1])
        path = args[1]

    files = []

    for path in args[1:]:
        files.extend(Search(path, L))

    Analyze(path, files)

Main()

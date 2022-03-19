import numpy as np

def ssort(elem):
    return elem[1]

lib = {}

with open("ClusterDesign_Step1_Results.txt", "r") as data:
    lines = data.readlines()

for line in lines:
    dat = [s for s in line.split('\t') if s]
    lib[int(dat[0].strip())] = [int(dat[0].strip()),float(dat[1].strip()),dat[2].strip(),dat[3].strip()]

clusters = {}

with open("groups.txt", "r") as data:
    lines = data.readlines()

cluster = []

i = 0

for line in lines:
    if line[0] == '>':
        if len(cluster) > 0:
            clusters[i] = {'seqs':cluster}
            i += 1
        cluster = []
    elif len(line.strip()) == 0: continue
    else: cluster.append(lib[int(line.strip())])

clusters[i] = {'seqs':cluster}

for i,c in clusters.items():
    best = min(c['seqs'], key=ssort)
    c['score'] = best[1]
    c['best'] = best[0]
    c['restraintmap'] = best[3]

clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: item[1]['score'])}

for i,c in clusters.items():
    if len(c['seqs']) == 1: continue
    print('#cluster id: ',i,' | best seq num: ', c['best'],' | best score: ', c['score'],' | seqs in cluster: ',len(c['seqs']))
    print('#restraintmap',c['restraintmap'])
    for s in c['seqs']:
        print('>' + str(s[0]), s[1])
        print(s[2])

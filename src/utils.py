"""Helpful utilities.

Many of these are based on the work by @lucidrains:
https://github.com/lucidrains/tr-rosetta-pytorch/blob/main/tr_rosetta_pytorch/utils.py
"""

# native
from pathlib import Path
import string

# lib
from matplotlib import pylab as plt
import numpy as np
import torch
import math
import random

# pkg
import config as cfg

def d(tensor=None, force_cpu=cfg.FORCECPU):
    """Return 'cpu' or 'cuda' depending on context. Is used to set tensor calculation device"""
    if force_cpu:
        return "cpu"
    if tensor is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if tensor.is_cuda else "cpu"


def distance_to_bin_id(angstrom_distance: float):
    """Return the `bin_id` for a distance from a structure prediction."""
    # Given a single (float) distance in Angstroms,
    # return the corresponding bin_id from trRosetta['dist'] prediction
    return (np.abs(cfg.bin_dict_np["dist"] - angstrom_distance)).argmin()

def average_dict(list_of_dicts, detach = False):
    """Returns a dict where each entry contains the average of the tensors for that key"""
    averaged_outputs = {}
    for key in list_of_dicts[0].keys():
        key_values = []
        for dict_el in list_of_dicts:
            key_values.append(dict_el[key])

        averaged_outputs[key] = torch.stack(key_values).mean(axis=0)

        if detach:
            averaged_outputs[key] = averaged_outputs[key].cpu().detach().numpy()
            torch.cuda.empty_cache()

    return averaged_outputs

def parse_a3m(filename):
    """Return the contents of an `.a3m` file as integers.

    The resulting integers are in the range 0..max_aa.
    """
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # TODO: add support for multi-line sequences
    # read file line by line
    for line in Path(filename).open():
        # skip labels
        if line[0] != ">":
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i in range(cfg.ALPHABET_full.shape[0]):
        msa[msa == cfg.ALPHABET_full[i]] = i

    # treat all unknown characters as gaps
    msa[msa > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return msa

#FFT
def parse_a3mseq(seq):
    seqs = [seq]
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # convert letters into numbers
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i in range(cfg.ALPHABET_full.shape[0]):
        msa[msa == cfg.ALPHABET_full[i]] = i

    # treat all unknown characters as gaps
    msa[msa > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return msa

def definegroupbydist(motifs, motif_npz_path, mode = 4):
    template = dict(np.load(motif_npz_path))["dist"] #distancemap

    distmat = np.zeros((len(motifs),len(motifs),2)) #matrix motif x motif [sum,count]

    k = 0
    for m1 in motifs[:]:
        r1 = range(0,m1[2])
        if mode == 5: r1 = range(m1[2]-1,m1[2]) #use only terminal residues?
        for i in r1: #local index
            s1i = m1[0]+i  #template index
            #c1 = m1[3][i]
            #if c1 == 's' or c1 == 'b': #contraint is structural?
            l = 0
            for m2 in motifs[:]:
                r2 = range(0,m2[2])
                if mode == 5: r2 = range(0,1) #use only terminal residues?
                for j in r2:
                    s2i = m2[0]+j  #template index
                    #c2 = m2[3][j]
                    #if c2 == 's' or c2 == 'b': #contraint is structural?
                    dist = template[s1i,s2i]
                    if dist > 0:
                        distmat[k,l,0] = distmat[k,l,0] + dist #summed distance
                        distmat[k,l,1] = distmat[k,l,1] + 1 #count number of distances
                l = l + 1

        for i in range(0,len(motifs)):
            if distmat[k,i,1] > 0.5: distmat[k,i,0] = distmat[k,i,0]/distmat[k,i,1] #calculate average
            else: distmat[k,i,0] = 30 #fix no contacts

        k = k + 1

    bestorder = list(range(0,len(motifs)))
    order = bestorder.copy()
    bestscore = 99999999999999
    for i in range(0,80000):
        random.shuffle(order)
        score = motifdistscore(distmat, order, mode)
        if score < bestscore:
            bestorder = order.copy()
            bestscore = score
        else: order = bestorder.copy()

    print(bestscore)
    print(bestorder)

    p = 0
    for i in bestorder:
        if p < len(bestorder) - 1:
            print(distmat[bestorder[p],bestorder[p+1],0])
        motifs[i][4] = p
        p = p + 1

    return motifs

def motifdistscore(distmat, order, mode = 4):
    score = 0
    for i in range(0,len(order) - 1):
        dist = distmat[order[i],order[i+1],0]
        score = score + dist*dist
        if mode == 4 and i < len(order)-2:
            dist = distmat[order[i],order[i+2],0]
            score = score + dist

    return score

def seqfrommotifs(motifs,sequence, bkgseq):
    newseq = ""
    for pos in range(len(bkgseq)):
        use_bkg = True
        for m in motifs[:]:
            if pos >= m[5] and pos <= m[6]:
                mpos = pos - m[5]
                if m[3][mpos] == 'b' or m[3][mpos] == 's':
                    templatepos = m[0] + mpos
                    use_bkg = False
                    newseq += sequence[templatepos]
                break

        if use_bkg == True: newseq += bkgseq[pos]

    return newseq


def aa2idx(seq: str) -> np.ndarray:
    """Return the sequence of characters as a list of integers."""
    # convert letters into numbers
    abc = cfg.ALPHABET_full
    idx = np.array(list(seq), dtype="|S1").view(np.uint8) #convert tto array of chars (byte ints)
    for i in range(abc.shape[0]):
        idx[idx == abc[i]] = i #for all equal to abs[i] set value to i (replace byte ints with 0 - 20)

    # treat all unknown characters as gaps
    idx[idx > cfg.MAX_AA_INDEX] = cfg.MAX_AA_INDEX

    return idx


def idx2aa(idx: np.ndarray) -> str:
    """Return the string representation from an array of integers."""
    abc = np.array(list(cfg.ALPHABET_core_str))
    return "".join(list(abc[idx]))


def sample_distogram_bins(distogram_distribution, n_samples):
    """Return a distogram sampled from a distogram distribution."""
    n_residues = distogram_distribution.shape[1]
    sampled_distogram_bins = np.zeros((n_residues, n_residues)).astype(int)

    for i in range(n_residues):
        for j in range(n_residues):
            prob = distogram_distribution[:, i, j]
            prob = prob / np.sum(prob)

            samples = np.random.choice(len(distogram_distribution), n_samples, p=prob)
            sampled_distogram_bins[i, j] = int(np.mean(samples))

    return sampled_distogram_bins


def distogram_distribution_to_distogram(
    distribution, reduction_style="max", keep_no_contact_bin=False
):
    """Return the estimated distogram from a distribution of distograms.

    NOTE: This function is specific to the trRosetta distogram format.
    """

    distribution = np.squeeze(distribution)
    if len(distribution.shape) > 3:
        raise "Error: distogram_distribution_to_distogram needs a single dist as input, not a batch!"

    # Remove only the special class "no-contact":
    if keep_no_contact_bin:
        distogram_distribution = distribution
    else:
        distogram_distribution = distribution[1:]

    # Bin Distances in Angstroms:
    distances = cfg.bin_dict_np["dist"][1:]

    if keep_no_contact_bin:
        distances = np.insert(distances, 0, 22, axis=0)

    if reduction_style == "max":
        D_pred_bins = np.argmax(distogram_distribution, axis=0)
    elif reduction_style == "mean":
        D_pred_bins = (
            np.abs(distogram_distribution - np.mean(distogram_distribution, axis=0))
        ).argmin(axis=0)
    elif reduction_style == "sample":
        D_pred_bins = sample_distogram_bins(distogram_distribution, 500)

    estimated_distogram = distances[D_pred_bins]
    np.fill_diagonal(estimated_distogram, 2)

    return estimated_distogram


### Plotting Function ###


def plot_distogram(distogram, savepath, title="", clim=None):
    """Save a plot of a distogram to the given path."""
    plt.imshow(distogram)
    plt.title(title, fontsize=14)
    plt.xlabel("Residue i")
    plt.ylabel("Residue j")
    if clim is not None:
        plt.clim(clim[0], clim[1])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close("all")


def plot_progress(E_list, savepath, title=""):
    """Save a plot of sequence losses to the given path."""
    x = np.array(range(len(E_list)))
    y = np.array(E_list)
    #mean = []
    #for i in range(0,len(x)):
    #    if i == 0: mean.append(y[0])
    #    elif i > 1000: mean.append(y[i]+np.std(y[i-1000:i]))
    #    else: mean.append(y[i]+np.std(y[0:i]))

    plt.plot(x, y, "o-")
    #plt.plot(x,mean,label = "std")
    plt.ylabel("Sequence Loss")
    plt.xlabel("N total attempted mutations")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close("all")

from matplotlib import pylab as plt
import numpy as np
import config as cfg
from pathlib import Path
import linecache
import sys
import random
from torch.cuda.amp import autocast
import mcmc
from tr_Rosetta_model import trRosettaEnsemble, prep_seq
from utils import aa2idx, distogram_distribution_to_distogram, idx2aa, plot_progress, d
import torch
from time

def distogram_distribution_to_distogram(distribution, reduction_style="max", keep_no_contact_bin=False):
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
        D_pred_bins = np.argmax(distogram_distribution, axis=2)
    elif reduction_style == "mean":
        D_pred_bins = ( np.abs(distogram_distribution - np.mean(distogram_distribution, axis=0)) ).argmin(axis=2)
    elif reduction_style == "sample":
        D_pred_bins = sample_distogram_bins(distogram_distribution, 500)

    estimated_distogram = distances[D_pred_bins]
    np.fill_diagonal(estimated_distogram, 2)

    return estimated_distogram

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

def get_sequence(L, aas, native_frequencies):
    if native_frequencies is None:
        native_frequencies = [1]*len(aas)


    seq = ""

    for i in range(L):
        seq = seq + random.choices(aas, native_frequencies, k=1)[0]

    return seq

def generate_bkg(length,n_samples,batchsize,met_first,rm_aa):
    met_first = met_first
    L = length
    n_samples = n_samples
    n_niter = int(n_samples / batchsize)
    
    print("Total samples: " + str(n_inter*batchsize))
    print("Iterations: " + str(n_iter))

    aas = list("ARNDCQEGHILKMFPSTWYV")
    for aa in rm_aa:
        if aa in aas:
            aas.remove(aa)
    sequences = []
    backgrounds = {'dist':[], 'omega':[], 'theta':[], 'phi':[]}

    nn = NN()

    startime = time.time()
    runtime = 0

    for j in range(n_iter):
        background = {'dist':[], 'omega':[], 'theta':[], 'phi':[]}

        for i in range(batchsize):
            print(f"Running iteration {j+1}, sample {i+1} of {n_samples}")
            seq = get_sequence(L, aas, cfg.native_freq)
            if met_first: seq = "M" + seq[1:]
            sequences.append(seq)
            seq = aa2idx(seq).copy().reshape([1, L])

            output = nn(seq)

            background['dist'].append(np.squeeze(output['dist']))
            background['theta'].append(np.squeeze(output['theta']))
            background['omega'].append(np.squeeze(output['omega']))
            background['phi'].append(np.squeeze(output['phi']))

        print("Averaging batch results...")
        for key in background.keys():
            print(key + "...")
            background[key] = torch.tensor(background[key])
            background[key] = torch.mean(background[key], axis=0).permute(2,1,0).numpy()
            backgrounds[key].append(background[key])
        print("Averaging done")
        runtime = time.time() - startime
        completed = (j+1)*batchsize
        waiting = n_samples - completed
        progress = completed / n_samples
        dt_sample = runtime/completed

        print(f"Expected finish in {dt_sample*waiting} seconds")


    print("Averaging all batches...")
    for key in background.keys():
        print(key + "...")
        backgrounds[key] = np.mean(backgrounds[key], axis=0)
    print("Averaging done")

    pref = ""
    if L < 100: pref = "0"
    np.savez_compressed(f'../backgrounds/background_distributions_{pref}{L}.npz', **backgrounds)
    print(f"Background generation completed for {L} residue sequences with {n_iter*n_samples} total samples")

def check_bkg(length):
    pref = ""
    if length < 100: pref = "0"
    bkg = dict(np.load(f'../backgrounds/background_distributions_{pref}{length}.npz'))
    distogram = distogram_distribution_to_distogram(bkg['dist'],'mean',True)
    plot_distogram(bkg['dist'][:,:,1],f'{pref}{length}_1.png','dist')
    plot_distogram(bkg['dist'][:,:,5],f'{pref}{length}_5.png','dist')
    plot_distogram(bkg['dist'][:,:,10],f'{pref}{length}_10.png','dist')
    plot_distogram(bkg['dist'][:,:,15],f'{pref}{length}_15.png','dist')
    plot_distogram(bkg['dist'][:,:,20],f'{pref}{length}_20.png','dist')
    plot_distogram(bkg['dist'][:,:,25],f'{pref}{length}_25.png','dist')
    plot_distogram(bkg['dist'][:,:,30],f'{pref}{length}_30.png','dist')
    plot_distogram(bkg['dist'][:,:,35],f'{pref}{length}_35.png','dist')

def main():
    L = 290

    #Generates background
    #L is background size. Larger L requires reduction of batchsize, depending on memory.
    #samples is the total number of sequences sampled for the bkg
    #batchsize is the number of sequences in each batch before memory is cleared
    #met_first, if true, first residue is Met
    #rm_aa is disallowed amino acids
    generate_bkg(length=L, n_samples=5000, batchsize = 50, met_first=True, rm_aa='C')
    
    #plots selected background distance probabilities
    #check_bkg(L)


class NN(torch.nn.Module):

    DEFAULT_ROOT_PATH = Path(__file__).parent.parent
    DEFAULT_MODEL_PATH = DEFAULT_ROOT_PATH / "models" / "trRosetta_models"
    DEFAULT_BGND_PATH = DEFAULT_ROOT_PATH / "backgrounds"
    DEFAULT_RESULTS_PATH = DEFAULT_ROOT_PATH / "results"

    def __init__(self):

        super().__init__()
        #self.results_dir = self.setup_results_dir(experiment_name)
        self.bkg_dir = "backgrounds"
        self.structure_models = trRosettaEnsemble("../models/trRosetta_models")  # .share_memory()
        print(f"{self.structure_models.n_models} structure prediction models loaded to {d()}")


    @torch.no_grad()
    def forward(self, seq):
        # Preprocess the sequence
        seq_curr = torch.from_numpy(seq).long()
        model_input, msa1hot = prep_seq(seq_curr)

        torch.cuda.empty_cache()
        with autocast():
            structure_predictions = self.structure_models(model_input, use_n_models=cfg.n_models, detach=True)

        return structure_predictions

if __name__ == "__main__":
    main()

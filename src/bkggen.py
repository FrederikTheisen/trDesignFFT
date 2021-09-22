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
        D_pred_bins = (
            np.abs(distogram_distribution - np.mean(distogram_distribution, axis=0))
        ).argmin(axis=2)
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

def plot_bkg():
    bkg = dict(np.load('../backgrounds/background_distributions_256.npz'))
    seq_L = 150
    for key in bkg:
        bkg[key] = bkg[key][:seq_L, :seq_L, :]
    distogram = distogram_distribution_to_distogram(bkg['dist'],'mean',True)
    plot_distogram(np.argmax(bkg['dist'][:,:,1:],axis=2),'tst.png','dist')

def get_sequence(L, aas, native_frequencies):
    if native_frequencies is None:
        native_frequencies = [1]*len(aas)


    seq = ""

    for i in range(L):
        seq = seq + random.choices(aas, native_frequencies, k=1)[0]

    return seq

def generate_bkg(length,n_samples,met_first):
    met_first = met_first
    L = length
    n_samples = n_samples

    aas = list("ARNDCQEGHILKMFPSTWYV")
    sequences = []
    background = {'dist':[], 'omega':[], 'theta':[], 'phi':[]}

    nn = NN()

    for i in range(n_samples):
        print(f"Running {i+1} of {n_samples}")
        seq = get_sequence(L, aas, cfg.native_freq)
        if met_first: seq = "M" + seq[1:]
        sequences.append(seq)
        seq = aa2idx(seq).copy().reshape([1, L])

        output = nn(seq)

        background['dist'].append(np.squeeze(output['dist']))
        background['theta'].append(np.squeeze(output['theta']))
        background['omega'].append(np.squeeze(output['omega']))
        background['phi'].append(np.squeeze(output['phi']))

    for key in background.keys():
        background[key] = torch.tensor(background[key])

    for key in background.keys():
        background[key] = torch.mean(background[key], axis=0).permute(2,1,0).numpy()

    pref = ""
    if L < 100: pref = "0"
    np.savez_compressed(f'../backgrounds/background_distributions_{pref}{L}.npz', **background)

def check_bkg(length):
    pref = ""
    if length < 100: pref = "0"
    bkg = dict(np.load(f'../backgrounds/background_distributions_{pref}{length}.npz'))
    distogram = distogram_distribution_to_distogram(bkg['dist'],'mean',True)
    plot_distogram(bkg['dist'][:,:,1],'tst.png','dist')

def main():
    L = 290

    generate_bkg(length=L,n_samples=180,met_first=True)

    check_bkg(L)


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

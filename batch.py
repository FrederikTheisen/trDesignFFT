#!/usr/bin/env python
"""Structure prediction.

Based on the implementation by @lucidrains:
https://github.com/lucidrains/tr-rosetta-pytorch/blob/main/tr_rosetta_pytorch/cli.py
"""

# native
from inspect import cleandoc
from pathlib import Path
import pathlib
import sys
from datetime import datetime

# lib
import numpy as np
import torch

#FFT imports
from torch.cuda.amp import autocast
import time
import gc
gc.collect()
torch.cuda.empty_cache()

# pkg
# pylint: disable=wrong-import-position
script_dir = Path(__file__).parent
sys.path[0:0] = [str(script_dir / "src"), str(script_dir)]

from tr_Rosetta_model import trRosettaEnsemble, preprocess, preprocessseq

import utils
from utils import parse_a3mseq

def setup_results_dir(experiment_name):
    """Create the directories for the results."""
    root = Path(__file__).parent.parent
    result_path = root / "batchpredict"
    results_dir = (
        result_path
        / experiment_name
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {results_dir}")
    return results_dir

@torch.no_grad()
def get_ensembled_predictions(ensemble_model, output_name=None, seq=None, index = None):
    """Use an ensemble of pre-trained networks to predict the structure of an MSA file."""
    start = time.time()

    input_data, _ = preprocessseq(seq)
    output_path = Path(str(setup_results_dir(output_name)) + "/" + str(index) + ".npz")
    output_fasta = Path(str(setup_results_dir(output_name)) + "/" + str(index) + ".fasta")

    # prob_distance, prob_omega, prob_theta, prob_phi
    torch.cuda.empty_cache()
    with autocast():
        outputs = [model(input_data) for model in ensemble_model.models]

    averaged_outputs = utils.average_dict(outputs, detach = True)

    end = time.time()
    print("NN runtime: " + str(end - start))

    np.savez_compressed(output_path, **averaged_outputs)
    with open(output_fasta,"w") as fasta:
        fasta.writelines([">\n",seq])

    print(f"predictions saved to {output_path}")

def main():
    with open("batchinput.txt") as reader:
        lines = reader.readlines()

    experiment_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    i = 0

    ensemble_model = trRosettaEnsemble()

    for line in lines:
        if len(line.strip()) > 2:
            get_ensembled_predictions(ensemble_model=ensemble_model, output_name=experiment_name, seq=line.strip(), index=i)
            i += 1


if __name__ == "__main__":
    main()

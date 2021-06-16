#!/usr/bin/env python
"""Structure prediction.

Based on the implementation by @lucidrains:
https://github.com/lucidrains/tr-rosetta-pytorch/blob/main/tr_rosetta_pytorch/cli.py
"""

# native
from inspect import cleandoc
from pathlib import Path
import sys

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

from tr_Rosetta_model import trRosettaEnsemble, preprocess
import utils

def get_ensembled_predictions(input_file, output_file=None, seq=None):
    """Use an ensemble of pre-trained networks to predict the structure of an MSA file."""
    ensemble_model = trRosettaEnsemble()

    start = time.time()

    if not seq:
        input_path = Path(input_file)
        input_data, _ = preprocess(msa_file=input_path)
    else:
        input_data, _ = preprocessseq(seq)

    output_path = (
        Path(output_file)
        if output_file
        else input_path.parent / f"{input_path.stem}.npz"
    )

    # prob_distance, prob_omega, prob_theta, prob_phi
    torch.cuda.empty_cache()
    with autocast():
        outputs = [model(input_data) for model in ensemble_model.models]

    averaged_outputs = utils.average_dict(outputs, detach = True)

    end = time.time()
    print("NN runtime: " + str(end - start))

    np.savez_compressed(output_path, **averaged_outputs)
    if seq is not None: print(f"predictions for {input_path} saved to {output_path}")
    else: print(f"predictions saved to {output_path}")

    #utils.plot_distogram(
    #    utils.distogram_distribution_to_distogram(averaged_outputs["dist"]),
    #    f"{input_file}_dist.jpg",
    #)


def main():
    """Predict structure using an ensemble of models.

    Usage: predict.py <input> [<output>]

    Options:
        <input>                 input `.a3m` or `.fasta` file
        <output>                output file (by default adds `.npz` to <input>)

    Examples:

    $ ./predict.py data/test.a3m
    $ ./predict.py data/test.fasta
    """
    useseq = False
    show_usage = False
    path = None
    args = sys.argv[1:]
    if len(args) == 1 and args[0] in ["-h", "--help"]:
        show_usage = True
    if len(args) > 2:
        i = 0
        for arg in args:
            if arg == "-seg":
                useseq = True
                seq = args[i+1]
            if arg == "-path":
                path = args[i+1].strip()
            i = i+1
    if not 1 <= len(args) <= 2:
        show_usage = True
        print("ERROR: Unknown number of arguments.\n\n")

    if show_usage:
        print(f"{cleandoc(main.__doc__)}\n")
        sys.exit(1)

    if not useseq:
        get_ensembled_predictions(*args)  # pylint: disable=no-value-for-parameter
    else:
        get_ensembled_predictions(args[0], seq=seq)


if __name__ == "__main__":
    main()

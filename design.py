#!/usr/bin/env python
"""Run the trDesign loop."""

# native
from pathlib import Path
import linecache
import sys

# lib
import numpy as np
from torch.cuda.amp import autocast

# pkg
# pylint: disable=wrong-import-position
script_dir = Path(__file__).parent
sys.path[0:0] = [str(script_dir / "src"), str(script_dir)]

import io
import mcmc
import utils
from utils import definegroupbydist
import config as cfg

def get_sequence(i, L, aa_valid, seed_file=None):
    """Return a sequence of length `L`.

    If `seed_file` is provided, return the first `L` characters of line `i`.
    Otherwise, return a completely random sequence using `aa_valid` symbols.
    """
    return (
        linecache.getline(seed_file, i + 1)[:L]
        if seed_file
        else utils.idx2aa(np.random.choice(aa_valid, L))
    )


def main():
    """Run the trDesign loop."""

    ########################################################
    # get valid residues
    ########################################################

    # any residue types to skip during sampling?
    aa_valid = np.arange(20)
    if cfg.RM_AA:
        aa_skip = utils.aa2idx(cfg.RM_AA.replace(",", ""))
        aa_valid = np.setdiff1d(aa_valid, aa_skip)

    ########################################################
    # prepare motifs
    ########################################################

    motifs = []
    mlen = 0

    if cfg.use_motifs:
        m_open = False
        motif = []
        for i in range(0, len(cfg.motif_constraint)):
            constraint = cfg.motif_constraint[i]
            group = cfg.motif_position[i]

            if group == '-':
                if m_open: #close motif and append
                    motifs.append(motif.copy())
                m_open = False
                continue
            elif not m_open:
                m_open = True
                motif = [i,i,0,constraint,int(group),0,0]
            elif m_open: #todo deal with no gap between two motifs
                motif[1] = i
                motif[3] = motif[3] + constraint

            if i == len(cfg.motif_constraint) - 1 and m_open: #close motif and append
                motifs.append(motif.copy())

        for m in motifs:
            l = m[1] - m[0] + 1
            m[2] = l
            mlen += l

        for m in motifs:
            print(m)

        print("Total Motif Length: " + str(mlen))

    else:
        motifs = None

    ########################################################
    # run MCMC
    ########################################################
    maxseqlen = cfg.LEN
    use_random_motif_mode = cfg.motif_placement_mode == -1

    if cfg.use_predef_start:
        Print("Using predefiend starting point")
        motifs = cfg.motifs
        cfg.sequence_constraint = cfg.b_seq_cn
        cfg.motif_placement_mode = -2
        cfg.LEN = len(cfg.best_seq)
        cfg.MCMC["BETA_START"] = 500

    seqs, seq_metrics = [], []
    for i in range(cfg.num_simulations):
        print("#####################################")
        print(f"\n --- Optimizing sequence {i:04} of {cfg.num_simulations:04}...")

        with open('control.txt', 'r') as reader:
            line = reader.readlines()[0].strip()
            if i > 0 and line == "exit":
                print("exiting due to command")
                break

        if cfg.use_random_length: #set random start length between length of motifs and config specified length
            cfg.LEN = np.random.randint(mlen+20, maxseqlen)

        if use_random_motif_mode:
            cfg.motif_placement_mode = np.random.randint(0,6)

        mcmc_optim = mcmc.MCMC_Optimizer(
            cfg.LEN,
            cfg.AA_WEIGHT,
            cfg.MCMC,
            cfg.native_freq,
            cfg.experiment_name,
            aa_valid,
            max_aa_index=cfg.MAX_AA_INDEX,
            sequence_constraint=cfg.sequence_constraint,
            target_motif_path=cfg.target_motif_path,
            motifs = motifs,
            motifmode = cfg.motif_placement_mode,
            motif_weight = np.random.uniform(1,cfg.motif_weight_max),
            bkg_weight = cfg.BKG_WEIGHT
        )

        start_seq = get_sequence(i, cfg.LEN, aa_valid, seed_file=cfg.seed_filepath)

        if cfg.use_predef_start:
            start_seq = cfg.best_seq

        with autocast():
            metrics = mcmc_optim.run(start_seq)

        seqs.append(metrics["sequence"])
        seq_metrics.append(metrics)

    with (script_dir / "metrics.csv").open("w") as f:
        i = 0
        f.write(f"num")
        for k,v in seq_metrics[0].items():
            f.write(f",{k}")
        f.write(f"\n")
        for itr in seq_metrics:
            f.write(str(i))
            for k,v in itr.items():
                f.write(f",{v}")
            f.write(f"\n")
            i += 1


if __name__ == "__main__":
    main()

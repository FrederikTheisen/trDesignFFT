#!/usr/bin/env python
"""Run the trDesign loop."""

# native
from pathlib import Path
import linecache
import sys
from datetime import datetime

# lib
import numpy as np
import random
from torch.cuda.amp import autocast

# pkg
# pylint: disable=wrong-import-position
script_dir = Path(__file__).parent
sys.path[0:0] = [str(script_dir / "src"), str(script_dir)]

import io
import mcmc
import utils
from utils import definegroupbydist, seqfrommotifs
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
                if motif[4] != int(group): #not same motif group. Close and open new.
                    motifs.append(motif.copy())
                    motif = [i,i,0,constraint,int(group),0,0]

                motif[1] = i
                motif[3] = motif[3] + constraint

            if i == len(cfg.motif_constraint) - 1 and m_open: #at end of sequence, close motif and append
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
        print("Using predefiend starting point")
        motifs = cfg.motifs
        cfg.sequence_constraint = cfg.best_seq
        cfg.motif_placement_mode = -2
        cfg.LEN = len(cfg.best_seq)
        cfg.MCMC["BETA_START"] = 200
    elif cfg.use_predef_motif:
        print("Using predefiend motifs")
        motifs = cfg.motifs
        cfg.motif_placement_mode = -3

    if mlen > 256:
        cfg.BACKGROUND = False


    seqs, seq_metrics = [], []
    for i in range(cfg.num_simulations):
        print("#####################################")
        print(f"\n --- Optimizing sequence {i:04} of {cfg.num_simulations:04}...")

        with open('control.txt', 'r') as reader:
            line = reader.readlines()[0].strip()
            if i > 0 and line == "exit":
                print("Exiting due to command")
                break

        if cfg.use_random_length: #set random start length between length of motifs and config specified length
            cfg.LEN = np.random.randint(mlen, maxseqlen)

        if use_random_motif_mode:
            cfg.motif_placement_mode = np.random.randint(0,6)

        if i > 0:
            cfg.TEMPLATE = random.choice([True, False])
            cfg.GRADIENT = random.choice([True, False])
            #cfg.MATRIX = random.choice([True, False])

        print("GRADIENT: " + str(cfg.GRADIENT))
        print("MATRIX: " + str(cfg.MATRIX))
        print("TEMPLATE: " + str(cfg.TEMPLATE))

        mtf_weight = cfg.motif_weight_max
        if cfg.use_random_motif_weight: mtf_weight = np.random.uniform(1,cfg.motif_weight_max)

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
            motif_weight = mtf_weight,
            bkg_weight = cfg.BKG_WEIGHT
        )

        start_seq = get_sequence(i, cfg.LEN, aa_valid, seed_file=cfg.seed_filepath)

        if cfg.use_predef_start:
            start_seq = cfg.best_seq
        elif cfg.TEMPLATE:
            start_seq = seqfrommotifs(motifs,cfg.sequence_constraint,start_seq)

        if cfg.first_residue_met:
            start_seq = "M" + start_seq[1:]

        metrics = mcmc_optim.run(start_seq)


        metrics["GRADIENT"] = str(cfg.GRADIENT)
        metrics["TEMPLATE"] = str(cfg.seq_from_template)
        metrics["MATRIX"] = str(cfg.MATRIX)
        seqs.append(metrics["sequence"])
        seq_metrics.append(metrics)

    output_file_name = "results/" + cfg.experiment_name + "/" + datetime.now().strftime("%Y-%m-%d-%H%M%S") + "_metrics" + ".csv"

    with (script_dir / output_file_name).open("w") as f:
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

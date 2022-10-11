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
    # prepare property template
    ########################################################

    if cfg.FILE_MSA is not None:
        with open(cfg.FILE_MSA) as reader:
            lines = reader.readlines()
            for line in lines:
                if len(line) > 10 and line[0:9] == 'BAR_GRAPH':
                    data = line.split('\t')
                    datatype = data[1].strip()
                    if datatype == 'Conservation':
                        annotations = data[3].split('|')
                        cfg.TEMPLATE_AA_PROPERTIES = []
                        #11.0,*,hydrophobic !aliphatic !aromatic !charged !negative !polar !positive !proline !small !tiny,[ffe600]
                        for a in annotations[:-1]:
                            dat = a.strip().split(',')
                            p = []
                            aas = cfg.ALPHABET_core_str

                            if len(dat) == 4:
                                for g in dat[2].strip().split(' '): p.append(g.strip())

                                for g in p:
                                    removeiffound = False
                                    group = ""

                                    if g[0] == '!':
                                        removeiffound = True
                                        group = cfg.AA_PROPERTY_GROUPS[g[1:]]
                                    else: group = cfg.AA_PROPERTY_GROUPS[g]

                                    if removeiffound:
                                        for c in group:
                                            if c in aas: aas = aas.replace(c,'')
                                    else:
                                        _aas = aas
                                        for aa in _aas:
                                            if aa not in group: aas = aas.replace(aa,'')
                                    if len(aas) == 0: aas = cfg.ALPHABET_core_str

                            for aa in cfg.RM_AA:
                                if aa in aas: aas = aas.replace(aa,'')
                            props = []

                            for aa in aas: props.append(aa)

                            cfg.TEMPLATE_AA_PROPERTIES.append(props)
                        print("TEMPLATE_AA_PROPERTIES LENGTH: " + str(len(cfg.TEMPLATE_AA_PROPERTIES)))
                    elif datatype == 'Consensus':
                        cfg.TEMPLATE_AA_CONSENSUS = []
                        cfg.TEMPLATE_AA_CONSENSUS_PROBABILITIES = []
                        annotations = data[3].split('|')
                        for a in annotations[:-1]:
                            dat = a.strip().split(',')[2]
                            aas = []
                            p = []

                            for g in dat.split(';'):
                                prob = float(g.strip().split(' ')[1][0:-1])
                                if prob > 1: #minimum probability
                                    aas.append(g.strip()[0])
                                    p.append(float(g.strip().split(' ')[1][0:-1]))
                                elif len(aas) == 0:
                                    aas.append(g.strip()[0])
                                    p.append(float(g.strip().split(' ')[1][0:-1]))

                            cfg.TEMPLATE_AA_CONSENSUS.append(aas)
                            cfg.TEMPLATE_AA_CONSENSUS_PROBABILITIES.append(p)

                        print("TEMPLATE_AA_CONSENSUS LENGTH:  " + str(len(cfg.TEMPLATE_AA_CONSENSUS)))

    if cfg.FILE_PSSM is not None:
        with open(cfg.FILE_PSSM) as reader:
            lines = reader.readlines()
            cfg.PSSM = [0] * cfg.LEN
            pos = 0
            for line in lines:
                if len(line) < 20: continue #too little data to support PSSM, assume empty line
                data = line.strip().split(' ')
                weights = []

                for w in data: weights.append(float(w))

                cfg.PSSM[pos] = weights
                pos += 1

    ########################################################
    # run MCMC
    ########################################################
    maxseqlen = cfg.LEN
    use_random_motif_mode = cfg.motif_placement_mode == -1

    seqs, seq_metrics = [], []
    for i in range(cfg.num_simulations):
        print("#####################################")
        print(f"\n --- Optimizing sequence {i:04} of {cfg.num_simulations:04}...")

        with open('control.txt', 'r') as reader:
            line = reader.readlines()[0].strip()
            if i > 0 and (line == "exit" or line == 'end'):
                print("CMD DESIGN END")
                break

        if cfg.use_random_length: #set random start length between length of motifs and config specified length
            minlen = mlen
            if cfg.MINIMUM_LENGTH is not None: minlen = cfg.MINIMUM_LENGTH
            cfg.LEN = np.random.randint(minlen, maxseqlen)

        if use_random_motif_mode:
            cfg.motif_placement_mode = np.random.randint(0,6)

        #if i > 0: #uncomment to enable some randomness in each run
            #cfg.motif_placement_mode = random.choice([3.3,4])
            #cfg.TEMPLATE = random.choice([True, False])
            #cfg.TEMPLATE_MODE = random.choice(['predefined','msa'])
            #cfg.USE_WEIGHTED_IDX = random.choice(['reciprocal', False])
            #cfg.OPTIMIZER = random.choice(['msa_start'])
            #cfg.DYNAMIC_MOTIF_PLACEMENT = random.choice([True,False,False,False,False])

        mtf_weight = cfg.motif_weight_max
        if cfg.use_random_motif_weight: mtf_weight = np.random.uniform(1,cfg.motif_weight_max)

        if cfg.PREDEFINED_MOTIFS:
            motifs = random.choice(cfg.mmotifs)
            cfg.motif_placement_mode = -3

        print("INFO")
        print("  LENGTH:   " + str(cfg.LEN))
        print("  OUTPUT:   " + cfg.experiment_name)
        print("MOTIFS:     " + str(cfg.use_motifs))
        print("  MODE:     " + cfg.motif_placement_mode_dict[cfg.motif_placement_mode])
        print("  MOTIFS:   " + str(len(motifs)))
        print("  WEIGHT:   " + str(mtf_weight))
        print("  DYNAMIC:  " + str(cfg.DYNAMIC_MOTIF_PLACEMENT))
        print("IDX WEIGHT: " + str(cfg.USE_WEIGHTED_IDX))
        print("OPTIMIZER:  " + str(cfg.OPTIMIZER))
        print("  FILE MAT: " + str(cfg.FILE_MATRIX))
        print("  FILE PSSM:" + str(cfg.FILE_PSSM))
        print("TEMPLATE:   " + str(cfg.TEMPLATE))
        print("  MODE:     " + str(cfg.TEMPLATE_MODE))
        print("  FILE MSA: " + str(cfg.FILE_MSA))

        optim = cfg.OPTIMIZER

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

        if cfg.TEMPLATE:
            if 'msa' in cfg.TEMPLATE_MODE:
                consensus_sequence = ""
                for i in range(cfg.LEN):
                    if 'argmax' in cfg.TEMPLATE_MODE: aa = cfg.TEMPLATE_AA_CONSENSUS[i][0]
                    else: aa = random.choices(cfg.TEMPLATE_AA_CONSENSUS[i], cfg.TEMPLATE_AA_CONSENSUS_PROBABILITIES[i], k=1)[0]
                    consensus_sequence += aa
                start_seq = consensus_sequence
            elif cfg.TEMPLATE_MODE == 'predefined':
                start_seq = cfg.best_seq
            else:
                start_seq = seqfrommotifs(motifs,cfg.sequence_constraint,start_seq)

        if cfg.first_residue_met:
            start_seq = "M" + start_seq[1:]

        metrics, terminate_run = mcmc_optim.run(start_seq)

        metrics["FOLDER"] = str(mcmc_optim.folder)
        metrics["USE_WEIGHTED_IDX"] = str(cfg.USE_WEIGHTED_IDX)
        metrics["TEMPLATE"] = str(cfg.TEMPLATE)
        metrics["TEMPLATE_MODE"] = str(cfg.TEMPLATE_MODE)
        metrics["OPTIMIZER"] = str(optim)
        metrics["DYNAMIC_MOTIFS"] = str(cfg.DYNAMIC_MOTIF_PLACEMENT)
        seqs.append(metrics["sequence"])
        seq_metrics.append(metrics)

        if terminate_run: break

    print("FINISHED: saving output metrics file")

    output_file_name = "results/" + cfg.experiment_name + "/" + datetime.now().strftime("%Y-%m-%d-%H%M%S") + "_metrics" + ".csv"

    metricsfilenumber = 1

    while Path((script_dir / output_file_name)).is_file():
        output_file_name = "results/" + cfg.experiment_name + "/" + datetime.now().strftime("%Y-%m-%d-%H%M%S") + "_metrics_" + str(metricsfilenumber) + ".csv"
        metricsfilenumber += 1
        print("iterating output file number: " + str(metricsfilenumber))

    print("Saving " + output_file_name + "...")

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

    print("Save completed")

if __name__ == "__main__":
    main()
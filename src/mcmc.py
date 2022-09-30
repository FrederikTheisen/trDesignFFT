"""Markov Chain Monte Carlo for trDesign."""

# native
from datetime import datetime
import time

# lib
import numpy as np
from numpy.random import choice
import torch
from torch.cuda.amp import autocast
import math
import random
import time
import itertools

# pkg
from losses import *  # pylint: disable=wildcard-import, unused-wildcard-import
from tr_Rosetta_model import trRosettaEnsemble, prep_seq
from utils import aa2idx, distogram_distribution_to_distogram, idx2aa, plot_progress, softmax
import config as cfg
from utils import definegroupbydist, plot_muts


def v(torch_value):
    """Return a detached value, if possible."""
    try:
        return torch_value.cpu().detach().item()
    except Exception:
        return torch_value

def motifsort(elem):
    return elem[4]

def placemotifs(motifs, seq_L, sequence, mode = 0):
    """Randomly position discontinous motifs and check if valid. motif = [start, end, length, restraints, group, newstart, newend]"""
    print("Placing motifs...")
    valid = False
    i = 0
    sum = 0

    for m in motifs:
        sum += m[2]
    if (sum > seq_L):
        print("Sequence too short")
        return placemotifs(motifs[:-1], seq_L, sequence, mode)
    elif sum == seq_L:
        pos = 0
        mode = -3
        for m in motifs:
            m[5] = pos
            m[6] = m[5] + m[2] - 1
            pos = pos + m[2]

    while not valid:
        if mode == 0 or mode == 1:
            for m in motifs[:]:
                m[5] = np.random.randint(0,seq_L-m[2]+1)
                m[6] = m[5]+m[2]-1
        elif mode == -2:
            print("predefined motifs")
            seq_con = ""
            for i in range(0,seq_L):
                restraint = "-"
                for m in motifs:
                    if i >= m[5] and i <= m[6]:
                        mi = i - m[5] #local index
                        c = m[3][mi]  #con type
                        if c in cfg.sequence_restraint_letters:
                            restraint = sequence[i]
                        continue

                seq_con = seq_con + restraint

            return motifs, seq_con
        elif mode == -3:
            break;
        elif mode == 2:
            _mn = len(motifs)
            _sum = sum
            _spacing = (seq_L-_sum)/(1+_mn)
            pos = int(abs(np.random.normal(0,_spacing+1)))
            for m in motifs[:]:
                    m[5] = pos
                    m[6] = m[5]+m[2]-1

                    _sum -= m[2]
                    _mn -= 1
                    _spacing = (seq_L-_sum-m[6])/(1+_mn)
                    #print(_sum,_mn,_spacing)
                    buffer = int(np.random.normal(_spacing,abs(_spacing)))

                    pos = m[6] + buffer

                    pos = np.clip(pos,0,seq_L-1)
        elif mode == 2.1:
            spacing = int((seq_L-sum-6)/(len(motifs)-1)) #calculate spacing for even placement with short unrestrained tails
            pos = int(seq_L-sum-((len(motifs)-1)*spacing)) #center the motifs
            for m in motifs[:]:
                    m[5] = pos
                    m[6] = m[5]+m[2]-1
                    pos = m[6]+spacing
        elif mode == 2.2:
            motifs[0][5] = 0
            motifs[0][6] = motifs[0][2]-1
            motifs[-1][6] = seq_L - 1
            motifs[-1][5] = seq_L - motifs[-1][2]
            spacing = int((seq_L-sum)/(len(motifs)-1))
            pos = motifs[0][6]
            for m in motifs[1:-1]:
                buffer = int(abs(np.random.normal(spacing,spacing)))
                pos += 1 + buffer
                m[5] = pos
                m[6] = m[5] + m[2] - 1
                pos = m[6]
        elif mode == 2.3:
            motifs[0][5] = 0
            motifs[0][6] = motifs[0][2]-1
            motifs[-1][6] = seq_L - 1
            motifs[-1][5] = seq_L - motifs[-1][2]
            spacing = int((seq_L-sum)/(len(motifs)-1))
            pos = motifs[0][6]
            for m in motifs[1:-1]:
                buffer = int(abs(np.random.uniform(0,2*spacing)))
                pos += 1 + buffer
                m[5] = pos
                m[6] = m[5] + m[2] - 1
                pos = m[6]
        elif mode == 3 or mode == 3.1 or mode == 3.2:
            motifs.sort(key=motifsort)
            for m in motifs[:]: #set all same group
                m[4] = 0
            return placemotifs(motifs, seq_L, sequence, mode = mode-1)
        elif mode == 4 or mode == 5:
            motifs = definegroupbydist(motifs,cfg.target_motif_path, mode)
            return placemotifs(motifs, seq_L, sequence, mode = 3)
        elif mode >= 6 and mode < 7:
            maxgroupnum = len(motifs)
            groupnum = random.randint(0, maxgroupnum)
            for m in motifs:
                m[4] = groupnum
                groupnum += 1
                if groupnum >= maxgroupnum: groupnum = 0
            return placemotifs(motifs, seq_L, sequence, mode = mode-3)
        else: return placemotifs(motifs, seq_L, sequence, mode = 1)

        #check if motifs are valid
        valid = True
        i = i + 1
        for m1 in motifs[:]:
            if m1[6] > seq_L - 1:
                print("ends outside")
                valid = False
                continue
            for m2 in motifs[:]:
                if m1 == m2: continue
                if (m1[5] >= m2[5]) and (m1[5] <= m2[6]):
                    valid = False
                    break
                elif m1[6] >= m2[5] and m1[6] <= m2[6]:
                    valid = False
                    break

        if i > 20000: #if not valid, place motifs manually
            print("No valid motif placements found, attempting sequential positions")
            random.shuffle(motifs)
            rest = seq_L - sum
            buffer = math.floor(rest/(len(motifs)+1))
            pos = buffer
            for m in motifs:
                m[5] = pos
                m[6] = m[5] + m[2]-1
                pos += m[2] + buffer
            break

    seq_con = ""

    constrain_seq = False
    if sequence is not None:
        for i in range(0,seq_L):
            restraint = "-"
            for m in motifs:
                if i >= m[5] and i <= m[6]:
                    mi = i - m[5] #local index
                    c = m[3][mi]  #con type
                    if c in cfg.sequence_restraint_letters:
                        si = m[0]+mi  #template index
                        restraint = sequence[si]
                        constrain_seq = True
                    continue

            if i == 0 and cfg.first_residue_met:
                restraint = "M"

            seq_con = seq_con + restraint

    if not constrain_seq:
        seq_con = None

    return motifs, seq_con

def createmask(motifs, seq_L, save_dir, is_site_mask = False, print = True):
    """Create mask for discontinous motifs"""

    #setup mask size
    motif_mask_g = np.zeros((seq_L, seq_L))
    motif_mask = np.zeros((seq_L, seq_L))

    for m1 in motifs[:]:
        for i in range(m1[5],m1[6]+1):
            c1 = m1[3][i-m1[5]]
            if c1 in cfg.structure_restraint_letters:  #contraint is structural?
                for m2 in motifs[:]:
                    if math.fabs(m2[4]-m1[4]) > 1: #motifs are not in restrained groups?
                        continue
                    for j in range(m2[5],m2[6]+1):
                        c2 = m2[3][j-m2[5]]
                        if c2 in cfg.structure_restraint_letters:
                            v1 = cfg.structure_restraint_mask_values[c1]
                            v2 = cfg.structure_restraint_mask_values[c2]
                            value = np.clip(v1*v2, 0, 10000)
                            motif_mask[i,j] = value
                            motif_mask_g[i,j] = value**0.5

    if not is_site_mask and print:
        plot_values = motif_mask_g.copy()
        plot_distogram(
            plot_values,
            save_dir / f"motif_mask_groups.jpg", clim = [0,max(cfg.structure_restraint_mask_values.values())]
        )

    return motif_mask

def mcopy(motifs):
    copy = []

    for m in motifs:
        copy.append(m.copy())

    return copy

class MCMC_Optimizer(torch.nn.Module):
    """Markov Chain Monte Carlo optimizer."""

    # We don't define `.forward()`, but it's nn-based.
    # pylint: disable=too-many-instance-attributes, abstract-method

    DEFAULT_ROOT_PATH = Path(__file__).parent.parent
    DEFAULT_MODEL_PATH = DEFAULT_ROOT_PATH / "models" / "trRosetta_models"
    DEFAULT_BGND_PATH = DEFAULT_ROOT_PATH / "backgrounds"
    DEFAULT_RESULTS_PATH = DEFAULT_ROOT_PATH / "results"

    def __init__(
        self,
        L,
        aa_weight,
        MCMC,
        native_frequencies,
        experiment_name,
        aa_valid,
        max_aa_index=20,
        sequence_constraint=None,
        target_motif_path=None,
        trRosetta_model_dir="models/trRosetta_models",
        background_distribution_dir="backgrounds",
        motifs=None,
        motifmode = 0,
        motif_weight = 1,
        bkg_weight = 1
    ):
        """Construct the optimizer."""
        super().__init__()
        self.results_dir = self.setup_results_dir(experiment_name)
        self.bkg_dir = background_distribution_dir
        self.structure_models = trRosettaEnsemble(trRosetta_model_dir)  # .share_memory()
        print(f"{self.structure_models.n_models} structure prediction models loaded to {d()}")

        # General params:
        self.eps = 1e-7
        self.seq_L = L
        self.motifs = None
        self.motifmode = motifmode
        self.use_sites = cfg.use_sites
        if motifs is not None:
            _motifs,_seq_con = placemotifs(motifs, self.seq_L, sequence_constraint, mode=self.motifmode)
            self.motifs = _motifs
            if _seq_con is not None: sequence_constraint = _seq_con

        for m in self.motifs: print(m)
        print(sequence_constraint)

        # Setup MCMC params:
        self.beta, self.N, self.coef, self.M, self.MaxM, self.badF, self.design_t_limit, self.design_starttime  = (
            MCMC["BETA_START"],
            MCMC["N_STEPS"],
            MCMC["COEF"],
            MCMC["M"],
            MCMC["MAX"],
            MCMC["BAD"],
            MCMC["T_LIMIT"],
            MCMC["T_START"],
        )
        self.timelimited = self.design_t_limit > 0

        self.aa_weight = aa_weight

        # Setup sequence constraints:
        self.aa_valid = aa_valid
        self.native_frequencies = native_frequencies

        self.seq_constraint = sequence_constraint
        if self.seq_constraint is not None:
            assert len(self.seq_constraint) == self.seq_L, \
            "Constraint length (%d) must == Seq_L (%d)" %(len(self.seq_constraint), self.seq_L)

            self.seq_constraint = (aa2idx(self.seq_constraint).copy().reshape([1, self.seq_L]))
            self.seq_constraint_indices = np.where(self.seq_constraint != max_aa_index, 1, 0)

        self.target_motif_path = target_motif_path
        self.setup_losses()

        # stats
        self.bad_accepts = []
        self.n_accepted_mutations = 0
        self.n_accepted_bad_mutations = 0
        self.best_metrics = {}
        self.best_step = 0
        self.best_sequence = None
        self.best_E = None
        self.step = 0
        self.motif_weight = motif_weight
        self.bkg_weight = bkg_weight

        self.motif_update_evaluator = Motif_Search_Satisfaction(self.target_motif_path, self.seq_L)

        self.matrix_setup()

    def matrix_setup(self):
        self.substitution_matrix = {}
        with open("src/" + cfg.FILE_MATRIX) as reader:
            columns = reader.readline().strip().split()
            lines = reader.readlines()
            for aa in columns:
                if aa not in cfg.ALPHABET_core_str: continue
                self.substitution_matrix[aa] = {}


            for line in lines:
                data = line.strip().split()
                aa1 = data[0]
                if aa1 not in cfg.ALPHABET_core_str: continue
                sub = {}
                for i in range(0,len(columns)):
                    aa2 = columns[i]
                    if aa2 not in cfg.ALPHABET_core_str: continue
                    value = data[i+1]
                    try: sub[aa2] = float(value)
                    except: sub[aa2] = 0

                self.substitution_matrix[aa1] = dict(sorted(sub.items(), key=lambda x: x[1]))

        if False: #convert to non negative values
            min_value = 1

            for a in self.substitution_matrix.items():
                for b in a[1].items():
                    if b[1] < min_value: min_value = b[1]

            if min_value < 0:
                offset = -min_value + 1

                for a in self.substitution_matrix.items():
                    for b in a[1].items():
                        self.substitution_matrix[a[0]][b[0]] += offset

        if 'gd' in cfg.OPTIMIZER:
            if 'pssm' in cfg.OPTIMIZER:
                print("LOADING PSSM WEIGHTS...")
                self.aa_weight_matrix = cfg.PSSM
            else: self.aa_weight_matrix = np.zeros([290,20]) #self.aa_weight_matrix = np.random.uniform(-1,1,[self.seq_L, 20])

            open(self.results_dir / "gradient.txt","w+")

    def setup_results_dir(self, experiment_name):
        """Create the directories for the results."""
        self.folder = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        results_dir = (
            self.DEFAULT_RESULTS_PATH
            / experiment_name
            / self.folder
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "distogram_evolution").mkdir(parents=True, exist_ok=True)
        #print(f"Writing results to {results_dir}")
        return results_dir

    def setup_losses(self):
        """Prepare the loss functions."""

        # Initialize protein background distributions:
        if cfg.BACKGROUND: self.bkg_loss = Structural_Background_Loss(self.seq_L, self.bkg_dir)
        self.aa_bkgr_distribution = torch.from_numpy(self.native_frequencies).to(d())

        # Motif-Loss:
        self.motif_weight = 1.00 #Multiplier for the loss contribution
        self.motif_mask = np.zeros((self.seq_L, self.seq_L)) #LxL tensor, int64, L currently specified in config.py. Cotains only zeros
        self.motif_mask = torch.from_numpy(self.motif_mask).long().to(d())

        #If motif target is specified. Motif target file is .npz LxL. L is overall protein length
        #Overrides previous motif_mask
        #New mask will be ones with a diagonal zero line
        if self.target_motif_path is not None:
            self.motif_mask = np.ones((self.seq_L, self.seq_L))

            if self.motifs is not None:
                self.motif_mask = createmask(self.motifs, self.seq_L, self.results_dir)

            self.motif_mask = torch.from_numpy(self.motif_mask).long().to(d())
            self.motif_mask.fill_diagonal_(0)
            self.motif_sat_loss = Motif_Satisfaction(self.target_motif_path, mask=self.motif_mask, save_dir=self.results_dir, motifs = self.motifs)

            self.site_weight = 0
            if self.use_sites:
                self.site_mask = createmask(self.motifs, self.seq_L, self.results_dir, is_site_mask = True)
                self.site_mask = torch.from_numpy(self.site_mask).long().to(d())
                self.site_mask.fill_diagonal_(0)

                self.site_sat_loss = Site_Satisfaction(
                    self.target_motif_path, mask=self.site_mask, motifs = self.motifs
                )


        # Apply the background KL-loss only under the hallucination_mask == 1 region
        self.hallucination_mask = 1 - torch.clip(self.motif_mask, 0, 1)
        self.hallucination_mask.fill_diagonal_(0)

    def loss(self, sequence, structure_predictions, msa1hot, track=False):
        """Compute the loss function."""

        # Top-prob: extremily expensive operation
        #TM_score_proxy = top_prob(structure_predictions['dist'], verbose=False)
        #TM_score_proxy = TM_score_proxy[0]  # We're running with batch_size = 1

        # Background KL-loss:
        if cfg.BACKGROUND: background_loss = self.bkg_loss(structure_predictions, hallucination_mask=self.hallucination_mask)
        else: background_loss = torch.tensor(0)

        # aa composition loss
        aa_samp = (msa1hot[0, :, :20].sum(axis=0) / self.seq_L + self.eps) #take entire first sequence, all 1hot chars, and sum across length?)  # Get relative frequency for each AA
        aa_samp = (aa_samp / aa_samp.sum())  # Normalize to turn into distributions (possibly redundant)
        loss_aa = (aa_samp* torch.log(aa_samp / (self.aa_bkgr_distribution + self.eps) + self.eps)).sum()

        # Motif Loss:
        if self.target_motif_path is not None: motif_loss, motif_loss_pos = self.motif_sat_loss(structure_predictions)
        else: motif_loss = 0 #no target motif = no loss

        if self.use_sites: site_loss = self.site_sat_loss(structure_predictions)
        else: site_loss = torch.tensor(0)

        # total loss
        loss_v = (self.bkg_weight * background_loss + self.aa_weight * loss_aa + self.motif_weight * motif_loss + self.site_weight * site_loss)

        metrics = {}
        if track:
            metrics["aa_weight"] = self.aa_weight
            metrics["background_loss"] = background_loss
            metrics["motif_loss"] = motif_loss
            metrics["site_loss"] = site_loss
            metrics["total_loss"] = loss_v
            #metrics["TM_score_proxy"] = TM_score_proxy

        return loss_v, metrics

    def search_motif_positions(self, structure_predictions):
        """Search algorithm for placing motifs"""
        fixed_ends = True

        variable_motifs = self.motifs
        print(len(variable_motifs))
        fixed_motifs = []
        first_residue = 0
        last_residue = self.seq_L - 1
        if fixed_ends:
            first_residue = variable_motifs[0][6] + 1
            last_residue = variable_motifs[-1][5] - 1
            variable_motifs = variable_motifs[1:-1]
            fixed_motifs.append(self.motifs[0])
            fixed_motifs.append(self.motifs[-1])

        self.motif_update_evaluator.squeeze(structure_predictions)

        if len(variable_motifs) == 0: return
        elif len(variable_motifs) == 1:
            #just check all positions
            print("NA")
        else:
            total_motif_length = 0
            for m in variable_motifs: total_motif_length += m[2]

            #order_options = itertools.permutations(variable_motifs, len(variable_motifs))

            #place two first motif first
            m1 = variable_motifs[0]
            m2 = variable_motifs[1]

            range_m1 = [first_residue, last_residue - total_motif_length - 1]
            range_m2 = [first_residue + m1[2], last_residue - total_motif_length + m1[2] - 1]

            tmp_motifs = [[0],[1],[2],[3]]

            best_tmp_motifs = []
            best_score = 9999999.9

            tmp_motifs[0] = fixed_motifs[0]
            tmp_motifs[3] = fixed_motifs[1]

            print("ranges: ", range_m1, range_m2)

            for s1 in range(range_m1[0], range_m1[1] + 1):
                tmp_motifs[1] = [m1[0], m1[1], m1[2], m1[3], m1[4], s1, s1 + m1[2] - 1]
                for s2 in range(tmp_motifs[1][6] + 1, range_m2[1] + 1):
                    tmp_motifs[2] = [m2[0], m2[1], m2[2], m2[3], m2[4], s2, s2 + m2[2] - 1]
                    _score = self.get_motif_placement_score(tmp_motifs, structure_predictions)

                    if _score < best_score or best_tmp_motifs is None:
                        best_score = _score
                        best_tmp_motifs = [tmp_motifs[1], tmp_motifs[2]]
                        print("better: ", best_tmp_motifs[0][5], best_tmp_motifs[1][5])

            print("best:   ", best_tmp_motifs[0][5], best_tmp_motifs[1][5])

            if len(variable_motifs) > 2:
                for m in variable_motifs[2:]:
                    total_tmp_motif_length = 0
                    for _m in best_tmp_motifs: total_tmp_motif_length += _m[2]

                    missing_motifs_restraint_length = total_motif_length - total_tmp_motif_length

                    last_allowed_position = last_residue - missing_motifs_restraint_length
                    tmp_motifs.append([])
                    result = None
                    best_score = 9999999.9
                    for s in range(best_tmp_motifs[-1][6] + 1, last_allowed_position):
                        tmp_motifs[-1] = [m[0], m[1], m[2], m[3], m[4], s, s + m[2] - 1]

                        _score = self.get_motif_placement_score(tmp_motifs, structure_predictions)

                        if _score < best_score or result is None:
                            best_score = _score
                            result = tmp_motifs[-1]
                            print(result)

                    best_tmp_motifs.append(result)


            self.motifs = best_tmp_motifs
            if fixed_ends:
                self.motifs.insert(0, fixed_motifs[0])
                self.motifs.append(fixed_motifs[1])

            print(best_tmp_motifs)

            self.motif_mask = createmask(self.motifs, self.seq_L, self.results_dir)
            self.motif_mask = torch.from_numpy(self.motif_mask).long().to(d())
            self.motif_mask.fill_diagonal_(0)

            self.motif_sat_loss.update(mask=self.motif_mask, motifs = self.motifs)

            self.hallucination_mask = 1 - torch.clip(self.motif_mask, 0, 1)
            self.hallucination_mask.fill_diagonal_(0)

    def check_motif_validity(self, motifs):
        """check if motifs overlap"""
        for m1 in motifs[:]:
            if m1[5] < 0: return False
            if m1[6] > self.seq_L - 1: return False
            for m2 in motifs[:]:
                if m1 == m2: continue
                if (m1[5] >= m2[5]) and (m1[5] <= m2[6]): return False
                elif m1[6] >= m2[5] and m1[6] <= m2[6]: return False

        return True

    def get_motif_placement_score(self, motifs, structure_predictions):
        """Get motif score for search algorithm"""
        motif_mask = np.zeros((self.seq_L, self.seq_L))

        for m1 in motifs[:]:
            for i in range(m1[5],m1[6]+1):
                c1 = m1[3][i-m1[5]]
                if c1 in cfg.structure_restraint_letters:  #contraint is structural?
                    v1 = cfg.structure_restraint_mask_values[c1]
                    for m2 in motifs[:]:
                        if m1 == m2: continue
                        if math.fabs(m2[4]-m1[4]) > 1: continue #motifs are not in restrained groups?
                        for j in range(m2[5],m2[6]+1):
                            c2 = m2[3][j-m2[5]]
                            if c2 in cfg.structure_restraint_letters:
                                v2 = cfg.structure_restraint_mask_values[c2]
                                value = np.clip(v1*v2, 1, 200)
                                motif_mask[i,j] = value

        motif_mask = torch.from_numpy(motif_mask).long().to(d())
        motif_mask = motif_mask.fill_diagonal_(0)

        #evaluator = Motif_Search_Satisfaction(self.target_motif_path, mask=motif_mask, motifs = self.motifs) #borken af?
        self.motif_update_evaluator.update(motif_mask, motifs)

        return self.motif_update_evaluator()

    def metropolis(self, seq, seq_curr, E_curr, E, motifs, motifs_curr):
        """Compute the Metropolis criterion."""
        accepted = True
        mod = 1
        deltaE = E_curr - E

        if motifs != motifs_curr: mod = .3

        # Metropolis criterion
        if E_curr < E:  # Lower energy, replace!
            seq = np.copy(seq_curr)
            motifs = mcopy(motifs_curr)
            E = E_curr
        else:  # Higher energy, maybe replace..
            if torch.exp((E - E_curr) * self.beta * mod) > np.random.uniform():
                seq = np.copy(seq_curr)
                motifs = mcopy(motifs_curr)
                E = E_curr
                self.bad_accepts.append(1)
                self.n_accepted_bad_mutations += 1
            else:
                accepted = False
                if motifs != motifs_curr:
                    seq = self.update_motifs(motifs, seq)
                self.bad_accepts.append(0)

        self.register_mutation_fitness(accepted, deltaE)

        # Update the best sequence:
        if E_curr < self.best_E:
            self.best_E = E_curr
            self.best_sequence = idx2aa(seq_curr[0])
            self.best_step = self.step
            self.best_motifs = mcopy(motifs_curr)

        return seq, E, motifs

    def mutate(self, seq, skipweight = False):
        """Return a mutated version of the sequence."""
        seq_curr = np.copy(seq)

        # Introduce a random mutation using the allowed aa_types:
        if cfg.USE_WEIGHTED_IDX is not False and not skipweight:
            if cfg.USE_WEIGHTED_IDX == 'good':
                idx = random.choices(range(self.seq_L), self.idx_weights, k=1)[0]
            elif cfg.USE_WEIGHTED_IDX == 'reciprocal':
                w = np.where(self.seq_constraint_indices, [0]*self.seq_L, [1 / (v + 1) for v in self.mutations])[0]
                idx = random.choices(range(self.seq_L), w, k=1)[0]
            else: idx = np.random.randint(self.seq_L)
        else: idx = np.random.randint(self.seq_L)

        from_aa = idx2aa(seq_curr[0])[idx]

        #Perform mutation
        mutation_list = self.mutation_score[idx]

        if cfg.OPTIMIZER is not None:
            if len(mutation_list) == 0: mutation_list.append([from_aa, from_aa, 0, True, 0]) #Adds accepted dummy mutation

            ################################
            ### SUBSTITUTION MATRIX MODE ###
            if 'matrix' in cfg.OPTIMIZER:
                last_accepted = None
                verybad = ""

                for m in mutation_list:
                    if m[2] > 0.1: verybad += m[1]
                    if m[3]: last_accepted = m

                _tried = ""
                for m in reversed(mutation_list):
                    if m[4] < self.step - self.seq_L: break;
                    _tried += m[1]

                if last_accepted is not None and last_accepted[2] < 0.01: p = 1
                else: p = 0

                list_of_candidates = [k for k,v in self.substitution_matrix[last_accepted[p]].items()]
                probability_distribution = softmax([v for k,v in self.substitution_matrix[last_accepted[p]].items()], axis = 0)

                for i in range(20):
                    aa = random.choices(list_of_candidates, probability_distribution, k=1)[0]
                    if aa in _tried:
                        print("tried...",idx,_tried,aa)
                        continue
                    if aa not in verybad: break

                seq_curr[0, idx] = aa2idx(aa)

            ################################
            ### MSA MUTATION MATRIX MODE ###
            elif 'conprob' in cfg.OPTIMIZER:
                aas = cfg.TEMPLATE_AA_CONSENSUS[idx]
                if len(aas) > 1:
                    aa = random.choices(cfg.TEMPLATE_AA_CONSENSUS[idx], cfg.TEMPLATE_AA_CONSENSUS_PROBABILITIES[idx], k=1)[0]
                    seq_curr[0, idx] = aa2idx(aa)
                else: seq_curr[0, idx] = np.random.choice(self.aa_valid)
            elif 'msa' in cfg.OPTIMIZER:
                aa1 = np.random.choice(cfg.TEMPLATE_AA_PROPERTIES[idx])
                aa2 = random.choices(cfg.TEMPLATE_AA_CONSENSUS[idx], cfg.TEMPLATE_AA_CONSENSUS_PROBABILITIES[idx], k=1)[0]
                seq_curr[0, idx] = aa2idx(np.random.choice([aa1,aa2]))
            elif 'pssm' in cfg.OPTIMIZER:

                _tried = from_aa
                for m in reversed(mutation_list):
                    if m[4] < self.best_step: break;
                    _tried += m[1]

                if len(_tried) >= len(self.aa_valid): return self.mutate(seq, skipweight=True)
                i = 0
                while True:
                    SM = cfg.PSSM[idx]
                    aa = random.choices(range(20), SM, k=1)[0]
                    if aa not in self.aa_valid: aa = np.random.choice(self.aa_valid)
                    if idx2aa(aa) in _tried and i < 200:
                        print("tried...",idx,_tried,idx2aa(aa))
                        i += 1
                        continue
                    break

                seq_curr[0, idx] = aa

            ###########################
            ### MODE NOT RECOGNIZED ###
            else: seq_curr[0, idx] = np.random.choice(self.aa_valid)

        #######################
        ### NON MATRIX MODE ###
        else: seq_curr[0, idx] = np.random.choice(self.aa_valid)

        #################################
        ### CONTROL.TXT mut injection ###
        if self.mutation_injection is not None:
            mut = self.mutation_injection
            _idx = mut[1]
            _from_aa = mut[0]
            if from_aa == _from_aa or _from_aa == 'X':
                _idx
                _from_aa = idx2aa(seq_curr[0, _idx])
                if mut[2] == 'X': seq_curr[0, _idx] = np.random.choice(self.aa_valid)
                else: seq_curr[0, _idx] = aa2idx(mut[2])
                self.mutation_injection = None
                self.current_mutations.append([_idx, _from_aa, idx2aa(seq_curr[0])[_idx]])
                self.report_result = True

        to_aa = idx2aa(seq_curr[0])[idx]

        if self.seq_constraint is not None:  # Fix the constraint:
            seq_curr = np.where(self.seq_constraint_indices, self.seq_constraint, seq_curr)

        if np.equal(seq_curr, seq).all(): # If the mutation did not change anything, retry
            self.report_result = False
            return self.mutate(seq, skipweight = True)

        #Check if mutating already change residue and store mutation information. Probably some recursive stuff...
        for m in self.current_mutations:
            if m[0] == idx:
                return seq

        self.current_mutations.append([idx, from_aa, to_aa])

        return seq_curr

    def mutate_motifs(self, motifs, seq, replacementmode = 1):
        _motifs = mcopy(motifs)
        _seq = np.copy(seq)
        for _ in range(1000):
            dpos = random.choices([-1,1])[0]
            if dpos == 0: continue

            idx = np.random.randint(len(_motifs))
            m = _motifs[idx]

            if replacementmode == 1:
                if dpos < 0:
                    if m[5] + dpos < 0: continue
                    _seq[0] = np.delete(np.insert(_seq[0],m[6]+1,_seq[0][m[5]-1]),m[5]-1)
                else:
                    if m[6] + dpos >= self.seq_L: continue
                    _seq[0] = np.delete(np.insert(_seq[0],m[5],_seq[0][m[6]+1]),m[6]+2)
            elif replacementmode == 2:
                if dpos < 0:
                    if m[5] + dpos < 0 or m[6] >= self.seq_L-1: continue
                    _seq[0] = np.delete(np.insert(_seq[0],m[6]+1,_seq[0][m[6]]),m[5]-1)
                else:
                    if m[6] + dpos >= self.seq_L or m[5] <= 0: continue
                    _seq[0] = np.delete(np.insert(_seq[0],m[5],_seq[0][m[5]-1]),m[6]+2)


            m[5] += dpos
            m[6] += dpos

            if random.random() > .3 and self.check_motif_validity(_motifs):
                _seq = self.update_motifs(_motifs, _seq, verbose=True)
                return _motifs, _seq

        return motifs, self.mutate(seq) #failed, do normal mutation instead and return original motifs

    def update_sequence_constraint(self, motifs, seq, verbose):
        seq_con = ""

        constrain_seq = False
        if cfg.sequence_constraint is not None:
            for i in range(0,self.seq_L):
                restraint = "-"
                for m in motifs:
                    if i >= m[5] and i <= m[6]:
                        mi = i - m[5] #local index
                        c = m[3][mi]  #con type
                        if c in cfg.sequence_restraint_letters:
                            si = m[0]+mi  #template index
                            restraint = cfg.sequence_constraint[si]
                            constrain_seq = True
                        continue

                if i == 0 and cfg.first_residue_met:
                    restraint = "M"

                seq_con = seq_con + restraint

        if not constrain_seq:
            return seq

        if verbose: print(seq_con)

        self.seq_constraint = (aa2idx(seq_con).copy().reshape([1, self.seq_L]))
        self.seq_constraint_indices = np.where(self.seq_constraint != cfg.MAX_AA_INDEX, 1, 0)

        if self.seq_constraint is not None:  # Fix the constraint:
            seq = np.where(self.seq_constraint_indices, self.seq_constraint, seq)

        return seq

    def update_motifs(self, motifs, seq, verbose = False):
        self.motif_mask = createmask(motifs, self.seq_L, self.results_dir, print = False)
        self.motif_mask = torch.from_numpy(self.motif_mask).long().to(d())
        self.motif_mask.fill_diagonal_(0)

        self.motif_sat_loss.update(mask=self.motif_mask, motifs = motifs, print = False)

        self.hallucination_mask = 1 - torch.clip(self.motif_mask, 0, 1)
        self.hallucination_mask.fill_diagonal_(0)

        seq = self.update_sequence_constraint(motifs, seq, verbose)

        return seq

    def get_gradient(self, perturbation_matrix, ck):
        """Return the estimated descent direction for each position in the sequence"""

        plus_matrix = self.aa_weight_matrix + ck * perturbation_matrix
        minus_matrix = self.aa_weight_matrix - ck * perturbation_matrix

        E_plus = self.score_probability_matrix(plus_matrix)
        E_minus = self.score_probability_matrix(minus_matrix)

        return (E_plus - E_minus)/(2.*ck*perturbation_matrix)

    def score_probability_matrix(self, matrix, n = 10, models = 1):
        score = 0.0
        if 'argmax' in cfg.OPTIMIZER:
            n = 1
            models = 3

        for i in range(n):
            model_input, msa = self.get_modelinput_from_pssm(matrix)
            out = self.structure_models(model_input, use_n_models=models)
            E, _ = self.loss("", out, msa, track=False)
            score += E

        mean = score/n

        return mean.cpu().detach().numpy()

    def get_modelinput_from_pssm(self, matrix):
        seq = aa2idx("A"*self.seq_L).copy().reshape([1, self.seq_L])
        softmaxtrix = softmax(matrix)
        argmax = 'argmax' in cfg.OPTIMIZER
        for idx in range(self.seq_L):
            while True:
                if argmax: aaidx = np.argmax(softmaxtrix[idx])
                else: aaidx = np.random.choice(range(20),p=softmaxtrix[idx])
                if aaidx in self.aa_valid or argmax: break
            seq[0, idx] = aaidx

        if self.seq_constraint is not None:  # Fix the constraint:
            seq = np.where(self.seq_constraint_indices, self.seq_constraint, seq)

        return prep_seq(torch.from_numpy(seq).long())

    def mutate_gd(self, seq):
        """Return a mutated version of the sequence."""
        seq_curr = np.copy(seq)

        ak = 1/self.step
        ck = 1/(self.step**0.5) #np.random.choice([1/6,0.5]))
        #perturbation_matrix = np.random.uniform(-0.01,0.01,[self.seq_L, 20])
        perturbation_matrix = np.random.randint(0, 2, [self.seq_L, 20]) * 2 - 1 # -1 or 1

        gradient = self.get_gradient(perturbation_matrix, ck)

        self.aa_weight_matrix -= ak * gradient
        softmaxtrix = softmax(self.aa_weight_matrix)

        argmax = True #'argmax' in cfg.OPTIMIZER
        for idx in range(self.seq_L):
            while True:
                if argmax: aaidx = np.argmax(softmaxtrix[idx])
                else: aaidx = np.random.choice(range(20),p=softmaxtrix[idx])
                if aaidx in self.aa_valid or argmax: break
            seq_curr[0, idx] = aaidx

        if self.seq_constraint is not None:  # Fix the constraint:
            seq_curr = np.where(self.seq_constraint_indices, self.seq_constraint, seq_curr)

        # with open(self.results_dir / "gradient.txt", "a") as f:
        #     line = ""
        #     for v in softmaxtrix[1]:
        #         line += f"{v:.5f} "
        #     f.write(line.strip() + "\n")
        return seq_curr

    def select_mutation_options(self, matrix, threshold = 0, above = True):
        if above:
            return [k for k,v in matrix.items() if v >= threshold]
        else:
            return [k for k,v in matrix.items() if v <= threshold]

    def diff_to_weight(self, diff, increase = True):
        if increase: return 2 - 0.025*diff
        else: return 0.9 + 0.02 * diff

    def register_mutation_fitness(self, accepted, deltaE):
        if deltaE < -1: deltaE = 0
        else: deltaE = float(deltaE.cpu().detach().numpy())

        good = deltaE < 0

        for mut in self.current_mutations:
            self.mutations[mut[0]] += 1

            if good: self.good_accepts.append(1)
            else: self.good_accepts.append(0)

            #for i in range(mut[0]-5,mut[0]+6):
            #    if i >= 0 and i < self.seq_L:
            #        self.idx_weights[i] *= self.diff_to_weight(abs(mut[0]-i), increase = accepted)

            self.idx_weights[mut[0]] *= self.diff_to_weight(0, increase = accepted)

            self.mutation_log.append([mut[0], mut[1], mut[2], self.substitution_matrix[mut[1]][mut[2]], deltaE, accepted])

            if accepted: self.n_accepted_mutations += 1

            self.mutation_score[mut[0]].append([mut[1], mut[2], deltaE, accepted, self.step]) #from, to, result, accepted, when

            if self.report_result:
                print("result:  " + str(mut) + " " + str(deltaE) + " | " + str(accepted))

        #self.idx_weights = np.clip([i+0.01*(1-i) for i in self.idx_weights], 0.2, 20)

        newweights = []

        for i in range(self.seq_L):
            sum = 0
            count = 0
            for j in range(-1,2):
                if i+j > 0 and i+j < self.seq_L-1:
                    sum += self.idx_weights[i+j]
                    count += 1

            newweights.append(sum/count)

        self.idx_weights = newweights

        self.current_mutations = []
        self.report_result = False

    def fixup_MCMC(self, seq, motifs):
        """Dynamically adjust the metropolis beta parameter to improve performance."""
        if self.step - self.best_step > 3000:
            # No improvement for a long time, reload the best_sequence and decrease beta:
            print("#### RELOADING BEST SEQUENCE ####")
            self.best_step = self.step
            self.beta = self.beta / (self.coef ** 2)
            seq = torch.from_numpy(aa2idx(self.best_sequence).reshape([1, self.seq_L])).long()

            motifs = mcopy(self.best_motifs)
            seq = self.update_motifs(motifs, seq)
            seq = torch.from_numpy(aa2idx(self.best_sequence).reshape([1, self.seq_L])).long()
            self.mutation_score = [[]] * self.seq_L

        elif np.mean(self.bad_accepts[-200:]) < self.badF: # There has been some progress recently, but we're no longer accepting any bad mutations...
            self.beta = self.beta / self.coef
        else:
            self.beta = self.beta * self.coef

        self.beta = np.clip(self.beta, 5, self.MaxM)

        return seq, motifs

    @torch.no_grad()
    def run(self, start_seq):
        """Run the MCMC loop."""
        # pylint: disable=too-many-locals

        start_time = time.time()
        muttime = 0
        nntime = 0
        losstime = 0
        scoretime = 0
        misctime = 0

        terminate_run = False

        # initialize with given input sequence
        print("Initial seq: ", start_seq)
        seq = aa2idx(start_seq).copy().reshape([1, self.seq_L])
        motifs = mcopy(self.motifs) #self.motifs.copy()

        if self.seq_constraint is not None:  # Fix the constraint:
            seq = np.where(self.seq_constraint_indices, self.seq_constraint, seq)

        nsave = min(max(1, self.N // 20),50)
        E, E_tracker = np.inf, []
        self.bad_accepts = []
        self.good_accepts = []
        self.n_accepted_mutations = 0
        self.n_accepted_bad_mutations = 0
        self.best_metrics = {}
        self.best_step = 0
        self.best_sequence = start_seq
        self.best_motifs = self.motifs
        self.mut_is_motif_perturbation = False
        self.best_E = E
        self.best_bkg = 0
        self.best_mtf = 0
        self.best_distogram_distribution = None
        self.idx_weights = [1] * self.seq_L
        self.mutations = [0] * self.seq_L
        self.mutation_score = [[] for _ in range(self.seq_L)]
        self.report_result = False
        self.current_mutations = []
        self.mutation_log = []
        self.last_update = time.time()
        self.last_graph = time.time()
        self.mutation_injection = None

        # Main loop:
        for self.step in range(self.N + 1):

            # random mutation at random position
            mut_start = time.time()
            if self.step > 0:
                seq_curr = seq
                motifs_curr = mcopy(motifs)
                if cfg.DYNAMIC_MOTIF_PLACEMENT and random.random() < 1/(self.step**0.5): #Move motif? not so often later in mcmc
                    motifs_curr, seq_curr = self.mutate_motifs(motifs_curr, seq_curr, replacementmode=random.choices([1,2])[0])
                elif cfg.OPTIMIZER is not None and 'gd' in cfg.OPTIMIZER:
                    for _ in range(10): seq_curr = self.mutate_gd(seq_curr)
                elif cfg.OPTIMIZER is not None and 'niter' in cfg.OPTIMIZER:
                    args = [int(s) for s in cfg.OPTIMIZER.split('_') if s.isdigit()]
                    niter = args[0]
                    if len(args) > 1 and self.step > args[1]: niter = 1
                    for _ in range(niter): seq_curr = self.mutate(seq_curr)
                else: seq_curr = self.mutate(seq) #default mutation mode

                if cfg.OPTIMIZER is not None and 'start' in cfg.OPTIMIZER:
                    if self.step > 4*self.seq_L: cfg.OPTIMIZER = 'none'
            else:
                seq_curr = seq
                motifs_curr = mcopy(motifs)


            # Preprocess the sequence
            seq_curr = torch.from_numpy(seq_curr).long()
            model_input, msa1hot = prep_seq(seq_curr)
            muttime += time.time() - mut_start

            # probe effect of mutation
            nn_start = time.time()
            structure_predictions = self.structure_models(model_input, use_n_models=cfg.n_models) #run trRosettaEnsemble -> runs trrosetta n times
            nntime += time.time() - nn_start

            #loss functions
            loss_start = time.time()
            E_curr, metrics = self.loss(seq_curr, structure_predictions, msa1hot, track=True)
            losstime += time.time() - loss_start

            #scoring evaluation
            score_start = time.time()

            if E_curr < self.best_E:
                self.best_bkg = metrics["background_loss"]
                self.best_mtf = metrics["motif_loss"]
                self.best_site = metrics["site_loss"]
                self.best_distogram_distribution = structure_predictions['dist']

            seq, E, motifs = self.metropolis(seq, seq_curr, E_curr, E, motifs, motifs_curr)
            E_tracker.append(v(E))
            delta_step_best = self.step - self.best_step

            scoretime += time.time() - score_start

            misc_start = time.time()
            if time.time() - self.last_update > cfg.report_interval or self.step == 0:
                fps = self.step / (time.time() - start_time)
                background_loss = metrics["background_loss"].cpu().detach().numpy()
                mtf_loss = metrics["motif_loss"].cpu().detach().numpy()
                site_loss = metrics["site_loss"].cpu().detach().numpy()

                b_bkg = self.best_bkg.cpu().detach().numpy()
                b_mtf = self.best_mtf.cpu().detach().numpy()
                b_site = self.best_site.cpu().detach().numpy()

                self.last_update = time.time()

                print(
                    f"Step {self.step:4d} / {self.N:4d} ({delta_step_best}) || "
                    f"beta: {self.beta:.1f}, "
                    f"mutations/s: {fps:.2f}, "
                    f"bad/good_accepts: {np.sum(self.bad_accepts[-100:])}/{np.sum(self.good_accepts[-100:])}",
                    flush=True,
                )
                print(f"STATS      || loss: {E:.3f}, accepted mutations: good: {sum(self.good_accepts)} | bad: {sum(self.bad_accepts)}")
                print(f"BEST STATS || loss: {self.best_E:.3f}, bkg: {b_bkg:.3f}, mtf: {b_mtf:.3f}, site: {b_site:.3f}")

                #timings
                total_time = time.time() - start_time
                print(f"total time: {total_time:.1f}")
                print(f"mut time:   {muttime:.2f}\t| {muttime/(len(self.mutation_log)+1):.4f}")
                print(f"nn time:    {nntime:.1f}\t| {nntime/(1+self.step):.4f}")
                print(f"loss time:  {losstime:.2f}\t| {losstime/(1+self.step):.4f}")
                print(f"score time: {scoretime:.1f}\t| {scoretime/(1+self.step):.4f}")
                print(f"misc time:  {misctime:.2f}\t| {misctime/(1+self.step):.4f}")

                plot_muts(self.idx_weights, self.results_dir / "idx_weights.jpg")
                plot_muts(self.mutations, self.results_dir / "mutations.jpg")

                if time.time() - self.last_graph > 1.8*cfg.report_interval or self.step == 0:
                    self.last_graph = time.time()

                    distogram = distogram_distribution_to_distogram(
                       self.best_distogram_distribution.cpu().detach().numpy()
                    )
                    plot_distogram(
                       distogram,
                       self.results_dir
                       / "distogram_evolution"
                       / f"{self.step:06d}_{self.best_E:.4f}.jpg",
                       clim=cfg.limits["dist"],
                    )

                    plot_progress(
                        E_tracker,
                        self.results_dir / "progress.jpg",
                        title=f"Optimization curve after {self.step} steps",
                    )
                    print(f"\n--- Current best:\n{self.best_sequence}")

                    if self.timelimited:
                        design_runtime = datetime.now() - self.design_starttime
                        if design_runtime.hours > self.design_t_limit: 
                            terminate_run = True
                            print("Time limit reached, breaking run...")
                            break


            ### HANDLE CONTROL.TXT INPUT ###
            try:
                if (1 + self.step) % (cfg.report_interval//2) == 0:
                    with open('control.txt', 'r') as reader:
                        lines = reader.readlines()
                        line = lines[0].strip()
                    if line == "eval":
                        cmd = lines[1].strip()
                        print("EVALUATE")
                        print("cmd: " + cmd)
                        try:
                            eval(cmd.strip())
                        except:
                            print("ERROR")
                    if line == "exit" or line == "break":
                        print("CMD RUN END")
                        break
                    elif line == 'optimizer':
                        cfg.OPTIMIZER = lines[1].strip()
                        print("OPTIMIZER: " + str(cfg.OPTIMIZER))
                    elif line == 'idxweight':
                        if lines[1].strip() == 'good': cfg.USE_WEIGHTED_IDX = 'good'
                        elif lines[1].strip() == 'reciprocal': cfg.USE_WEIGHTED_IDX = 'reciprocal'
                        elif lines[1].strip() == 'no': cfg.USE_WEIGHTED_IDX = False
                        print("USE_WEIGHTED_IDX: " + str(cfg.USE_WEIGHTED_IDX))
                    elif line == "beta":
                        value = float(lines[1].strip())
                        self.beta = value
                    elif line == "bkw":
                        value = float(lines[1].strip())
                        self.bkg_weight = value
                    elif line == "aaw":
                        value = float(lines[1].strip())
                        self.aa_weight = value
                    elif line == "mw":
                        value = float(lines[1].strip())
                        self.motif_weight = value
                    elif line == 'mutate':
                        data = lines[1].strip()
                        input = data.split(' ')
                        aa1 = input[0]
                        pos = int(input[1])
                        aa2 = input[2]
                        self.mutation_injection = [aa1,pos,aa2]
                    elif line == "pause":
                        while True:
                            with open('control.txt', 'r') as reader:
                                line = reader.readlines()[0].strip()
                            if line != "pause":
                                break
                            else:
                                print("pause...")
                                time.sleep(30)
            except: print('control input error')

            if (delta_step_best > 300 and self.step > 1500 and self.step % 100 == 0) or (cfg.FAST and self.step > 499 and self.step % 50 == 0):
                std = np.std(np.array(E_tracker)[-1000:])
                n_std = std / np.array(E_tracker)[-1000:].mean()
                print("sd: " + str(std))
                if abs(std) < 0.005 or (cfg.FAST and abs(std) < 0.02):
                    break

            if self.step % self.M == 0 and self.step != 0:
                seq, motifs = self.fixup_MCMC(seq, motifs)

            misctime += time.time() - misc_start


                

        ########################################

        # Save final results before exiting:
        seq_curr = torch.from_numpy(aa2idx(self.best_sequence).reshape([1, self.seq_L])).long()
        model_input, msa1hot = prep_seq(seq_curr)
        structure_predictions = self.structure_models(model_input)
        E_curr, self.best_metrics = self.loss(seq_curr, structure_predictions, msa1hot, track=True)

        TM_score_proxy = top_prob(structure_predictions['dist'], verbose=False)
        TM_score_proxy = TM_score_proxy[0]  # We're running with batch_size = 1

        self.best_metrics["TM_score_proxy"] = TM_score_proxy
        self.best_metrics["sequence"] = self.best_sequence
        self.best_metrics["motifweight"] = self.motif_weight
        self.best_metrics["motifmode"] = self.motifmode
        self.best_metrics["steps"] = self.step
        self.best_metrics["motifs"] = "\"" + str(self.best_motifs.copy()) + "\""

        # Dump distogram:
        best_distogram_distribution = structure_predictions['dist'].detach().cpu().numpy()
        distogram = distogram_distribution_to_distogram(best_distogram_distribution)
        plot_distogram(distogram, self.results_dir / f"result.jpg", clim=cfg.limits["dist"])
        plot_progress(E_tracker, self.results_dir / "progress.jpg", title=f"Optimization curve after {self.step} steps")

        # Write results to csv:
        with (self.results_dir / "result.csv").open("w") as f:
            for key, val in self.best_metrics.items():
                f.write(f"{key},{val}\n")

        with (self.results_dir / "mutation_log.csv").open("w") as f:
            for mut in self.mutation_log:
                f.write(f"{mut}\n")


        return self.best_metrics, terminate_run

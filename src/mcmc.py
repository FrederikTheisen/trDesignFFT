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

# pkg
from losses import *  # pylint: disable=wildcard-import, unused-wildcard-import
from tr_Rosetta_model import trRosettaEnsemble, prep_seq
from utils import aa2idx, distogram_distribution_to_distogram, idx2aa, plot_progress
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
            spacing = (seq_L-sum)/(1+len(motifs))
            pos = int(spacing)
            for m in motifs[:]:
                    buffer = int(abs(np.random.normal(spacing,spacing)))
                    m[5] = pos
                    m[6] = m[5]+m[2]-1
                    pos = m[6] + buffer

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
            last = motifs[-1]
            spacing = int((seq_L-sum)/(len(motifs)-1))
            pos = motifs[0][6]
            for m in motifs[1:-1]:
                buffer = int(abs(np.random.normal(spacing,spacing)))
                pos += 1 + buffer
                m[5] = pos
                m[6] = m[5] + m[2] - 1
                pos = m[6]

        elif mode == 3:
            motifs.sort(key=motifsort)
            for m in motifs[:]: #set all same group
                m[4] = 0
            return placemotifs(motifs, seq_L, sequence, mode = 2)
        elif mode == 4 or mode == 5:
            motifs = definegroupbydist(motifs,cfg.target_motif_path, mode)
            return placemotifs(motifs, seq_L, sequence, mode = 3)
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


def createmask(motifs, seq_L, save_dir, is_site_mask = False):
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
                            value = np.clip(v1*v2, 1, 100)
                            motif_mask[i,j] = value
                            motif_mask_g[i,j] = value**0.5

    if not is_site_mask:
        plot_values = motif_mask_g.copy()
        plot_distogram(
            plot_values,
            save_dir / f"motif_mask_groups.jpg",
        )

    return motif_mask

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
        self.structure_models = trRosettaEnsemble(
            trRosetta_model_dir
        )  # .share_memory()
        print(
            f"{self.structure_models.n_models} structure prediction models loaded to {d()}"
        )

        # General params:
        self.eps = 1e-7
        self.seq_L = L
        self.motifs = None
        self.motifmode = motifmode
        self.use_sites = cfg.use_sites
        print("Sequence Length: " + str(self.seq_L))
        if motifs is not None:
            _motifs,_seq_con = placemotifs(motifs, self.seq_L, sequence_constraint, mode=self.motifmode)
            self.motifs = _motifs
            if _seq_con is not None: sequence_constraint = _seq_con

        for m in self.motifs: print(m)
        print(sequence_constraint)

        # Setup MCMC params:
        self.beta, self.N, self.coef, self.M, self.MaxM = (
            MCMC["BETA_START"],
            MCMC["N_STEPS"],
            MCMC["COEF"],
            MCMC["M"],
            MCMC["MAX"],
        )
        self.aa_weight = aa_weight

        # Setup sequence constraints:
        self.aa_valid = aa_valid
        self.native_frequencies = native_frequencies

        self.seq_constraint = sequence_constraint
        if self.seq_constraint is not None:
            assert len(self.seq_constraint) == self.seq_L, \
            "Constraint length (%d) must == Seq_L (%d)" %(len(self.seq_constraint), self.seq_L)

            self.seq_constraint = (
                aa2idx(self.seq_constraint).copy().reshape([1, self.seq_L])
            )
            self.seq_constraint_indices = np.where(
                self.seq_constraint != max_aa_index, 1, 0
            )

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

        self.matrix_setup()

    def matrix_setup(self):
        self.substitution_matrix = {}
        with open("src/" + cfg.MATRIXFILE) as reader:
            columns = reader.readline().strip().split()
            lines = reader.readlines()
            for aa in columns:
                if aa in cfg.RM_AA: continue
                if aa not in cfg.ALPHABET_core_str: continue
                self.substitution_matrix[aa] = {}


            for line in lines:
                data = line.strip().split()
                aa1 = data[0]
                if aa1 not in cfg.ALPHABET_core_str: continue
                sub = {}
                for i in range(0,len(columns)):
                    aa2 = columns[i]
                    if aa2 in cfg.RM_AA: continue
                    if aa2 not in cfg.ALPHABET_core_str: continue
                    value = data[i+1]
                    try: sub[aa2] = float(value)
                    except: sub[aa2] = 0

                self.substitution_matrix[aa1] = dict(sorted(sub.items(), key=lambda x: x[1]))

        if True: #convert to non negative values
            min_value = 1

            for a in self.substitution_matrix.items():
                for b in a[1].items():
                    if b[1] < min_value: min_value = b[1]

            if min_value < 0:
                offset = -min_value + 1

                for a in self.substitution_matrix.items():
                    for b in a[1].items():
                        self.substitution_matrix[a[0]][b[0]] += offset

                #for a in self.substitution_matrix.items():
                #    for b in a[1].items():
                #        self.substitution_matrix[a[0]][b[0]] = self.substitution_matrix[a[0]][b[0]]*self.substitution_matrix[a[0]][b[0]]



    def setup_results_dir(self, experiment_name):
        """Create the directories for the results."""
        results_dir = (
            self.DEFAULT_RESULTS_PATH
            / experiment_name
            / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "distogram_evolution").mkdir(parents=True, exist_ok=True)
        print(f"Writing results to {results_dir}")
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
            self.motif_sat_loss = Motif_Satisfaction(
                self.target_motif_path, mask=self.motif_mask, save_dir=self.results_dir, motifs = self.motifs
            )

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
        if cfg.BACKGROUND:
            background_loss = self.bkg_loss(
                structure_predictions, hallucination_mask=self.hallucination_mask
            )
        else:
            background_loss = torch.tensor(0)

        # aa composition loss
        aa_samp = (
            msa1hot[0, :, :20].sum(axis=0) / self.seq_L + self.eps #take entire first sequence, all 1hot chars, and sum across length?
        )  # Get relative frequency for each AA
        aa_samp = (
            aa_samp / aa_samp.sum()
        )  # Normalize to turn into distributions (possibly redundant)
        loss_aa = (
            aa_samp
            * torch.log(aa_samp / (self.aa_bkgr_distribution + self.eps) + self.eps)
        ).sum()

        # Motif Loss:
        if self.target_motif_path is not None: #check if target motif has been specified
            motif_loss, motif_loss_pos = self.motif_sat_loss(structure_predictions) #motif_sat_loss = MotifSatisfaction object (nn object)
        else:
            motif_loss = 0 #no target motif = no loss

        if self.use_sites:
            site_loss = self.site_sat_loss(structure_predictions)
        else:
            site_loss = torch.tensor(0)

        # total loss
        loss_v = (
            self.bkg_weight * background_loss + self.aa_weight * loss_aa + self.motif_weight * motif_loss + self.site_weight * site_loss
        )

        metrics = {}
        if track:
            metrics["aa_weight"] = self.aa_weight
            metrics["background_loss"] = background_loss
            metrics["motif_loss"] = motif_loss
            metrics["site_loss"] = site_loss
            metrics["total_loss"] = loss_v
            #metrics["TM_score_proxy"] = TM_score_proxy

        return loss_v, metrics

    def metropolis(self, seq, seq_curr, E_curr, E):
        """Compute the Metropolis criterion."""
        accepted = True
        deltaE = E_curr - E

        # Metropolis criterion
        if E_curr < E:  # Lower energy, replace!
            seq = np.copy(seq_curr)
            E = E_curr
        else:  # Higher energy, maybe replace..
            if torch.exp((E - E_curr) * self.beta) > np.random.uniform():
                seq = np.copy(seq_curr)
                E = E_curr
                self.bad_accepts.append(1)
                self.n_accepted_bad_mutations += 1
            else:
                accepted = False
                self.bad_accepts.append(0)

        self.register_mutation_fitness(accepted, deltaE)

        # Update the best sequence:
        if E_curr < self.best_E:
            self.best_E = E_curr
            self.best_sequence = idx2aa(seq_curr[0])
            self.best_step = self.step

        return seq, E

    def mutate(self, seq):
        """Return a mutated version of the sequence."""
        seq_curr = np.copy(seq)

        # Introduce a random mutation using the allowed aa_types:
        if cfg.GRADIENT: idx = random.choices(range(self.seq_L), self.gradient, k=1)[0]
        else: idx = np.random.randint(self.seq_L)

        from_aa = idx2aa(seq_curr[0])[idx]

        #Perform mutation
        if cfg.MATRIX and self.mutation_score[idx][0] is not None:
            mut = self.mutation_score[idx]
            mut_score = mut[2]
            accepted = mut[3]
            mins = min(self.mutation_score, key=lambda x: x[2])[2]
            maxs = max(self.mutation_score, key=lambda x: x[2])[2]

            if cfg.MATRIX_MODE == 'probability':
                if mut_score > 0.66 * maxs and accepted:
                    print("##Revert: " + str(idx) + " " + str(mut))
                    self.reverse_log = True
                    seq_curr[0, idx] = aa2idx(mut[0])
                else:
                    if mut_score > 0.33 * maxs:
                        p = 0
                        print("Bad mut:  " + str(idx) + " " + str(mut))
                        self.reverse_log = True
                    elif mut_score > 0.1 * maxs: p = 0
                    else: p = 1

                    list_of_candidates = [k for k,v in self.substitution_matrix[mut[p]].items()]
                    probability_distribution = [v for k,v in self.substitution_matrix[mut[p]].items()]
                    mutation = random.choices(list_of_candidates, probability_distribution, k=1)
                    seq_curr[0, idx] = aa2idx(mutation)
            elif cfg.MATRIX_MODE == 'groups':
                ref = ""
                if mut_score < 0: ref = mut[1]
                else: ref = mut[0]

                group = []
                for k,v in cfg.AA_PROPERTY_GROUPS.items():
                    if ref in v:
                        for aa in v:
                            if aa in cfg.RM_AA: continue
                            group.append(aa)

                seq_curr[0, idx] = aa2idx(np.random.choice(group))
            elif cfg.MATRIX_MODE == 'msa':
                group = cfg.TEMPLATE_AA_PROPERTIES[idx]
                seq_curr[0, idx] = aa2idx(np.random.choice(group))
            else: seq_curr[0, idx] = np.random.choice(self.aa_valid)

        elif self.mutation_score[idx][0] is not None:
            mut = self.mutation_score[idx]
            mut_score = mut[2]

            maxs = max(self.mutation_score, key=lambda x: x[2])[2]

            if mut_score > 0.66 * maxs and mut[3]:
                print("##Revert: " + str(idx) + " " + str(mut))
                self.reverse_log = True
                seq_curr[0, idx] = aa2idx(mut[0])
            else: seq_curr[0, idx] = np.random.choice(self.aa_valid)
        else: seq_curr[0, idx] = np.random.choice(self.aa_valid)

        to_aa = idx2aa(seq_curr[0])[idx]

        if self.seq_constraint is not None:  # Fix the constraint:
            seq_curr = np.where(
                self.seq_constraint_indices, self.seq_constraint, seq_curr
            )

        if np.equal(seq_curr, seq).all(): # If the mutation did not change anything, retry
            self.reverse_log = False
            return self.mutate(seq)

        # Store mutation information
        self.current_mutations.append([idx, from_aa, to_aa])

        return seq_curr

    def select_mutation_options(self, matrix, threshold = 0, above = True):
        if above:
            return [k for k,v in matrix.items() if v >= threshold]
        else:
            return [k for k,v in matrix.items() if v <= threshold]

    def diff_to_weight(self, diff, increase = True):
        if increase:
            if diff == 0: return 1.25
            else: return 1.5 - 0.025*diff
        else: return 0.8 + 0.02 * diff

    def register_mutation_fitness(self, accepted, deltaE):
        if deltaE < -1: deltaE = 0
        else: deltaE = float(deltaE.cpu().detach().numpy())

        good = deltaE < 0

        for mut in self.current_mutations:
            self.good_mutations[mut[0]] += 1

            if good:
                self.good_accepts.append(1)
                for i in range(mut[0]-20,mut[0]+21):
                    if i >= 0 and i < self.seq_L:
                        self.gradient[i] *= self.diff_to_weight(abs(mut[0]-i))
            else:
                self.good_accepts.append(0)
                for i in range(mut[0]-10,mut[0]+11):
                    if i >= 0 and i < self.seq_L:
                        self.gradient[i] *= self.diff_to_weight(abs(mut[0]-i), increase = False)

            self.mutation_log.append([mut[0], mut[1], mut[2], self.substitution_matrix[mut[1]][mut[2]], deltaE, accepted])

            if accepted:
                self.n_accepted_mutations += 1

            self.mutation_score[mut[0]] = [mut[1], mut[2], deltaE, accepted]

            if self.reverse_log:
                print("result:  " + str(mut) + " " + str(deltaE) + " | " + str(accepted))

        self.gradient = np.clip([i+0.01*(1-i) for i in self.gradient], 0.2, 20)
        self.current_mutations = []
        self.reverse_log = False

    def fixup_MCMC(self, seq):
        """Dynamically adjust the metropolis beta parameter to improve performance."""
        if self.step - self.best_step > min(self.N // 4,1000):
            # No improvement for a long time, reload the best_sequence and decrease beta:
            print("reload best seq")
            self.best_step = self.step
            self.beta = self.beta / (self.coef ** 2)
            seq = torch.from_numpy(
                aa2idx(self.best_sequence).reshape([1, self.seq_L])
            ).long()
            self.mutation_score = [[None,None,0,False]] * self.seq_L

        elif np.mean(self.bad_accepts[-100:]) < 0.03:
            # There has been some progress recently, but we're no longer accepting any bad mutations...
            self.beta = self.beta / self.coef
        else:
            self.beta = self.beta * self.coef

        self.beta = np.clip(self.beta, 5, self.MaxM)

        return seq

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

        # initialize with given input sequence
        print("Initial seq: ", start_seq)
        seq = aa2idx(start_seq).copy().reshape([1, self.seq_L])

        nsave = min(max(1, self.N // 20),50)
        E, E_tracker = np.inf, []
        self.bad_accepts = []
        self.good_accepts = []
        self.n_accepted_mutations = 0
        self.n_accepted_bad_mutations = 0
        self.best_metrics = {}
        self.best_step = 0
        self.best_sequence = start_seq
        self.best_E = E
        self.best_bkg = 0
        self.best_mtf = 0
        self.best_distogram_distribution = None
        self.gradient = [1] * self.seq_L
        self.good_mutations = [0] * self.seq_L
        self.mutation_score = [[None,None,0]] * self.seq_L
        self.reverse_log = False
        self.current_mutations = []
        self.mutation_log = []
        self.last_update = time.time()
        self.last_graph = time.time()

        # Main loop:
        for self.step in range(self.N + 1):

            # random mutation at random position, also fix sequence constraint
            mut_start = time.time()
            if self.step != 0:
                seq_curr = self.mutate(seq)
                if delta_step_best < 30:
                    for i in range(4): seq_curr = self.mutate(seq_curr)
            else: seq_curr = seq


            # Preprocess the sequence, relatively expensive operations (0.04s per pass)
            seq_curr = torch.from_numpy(seq_curr).long()
            model_input, msa1hot = prep_seq(seq_curr)
            muttime += time.time() - mut_start


            # probe effect of mutation
            nn_start = time.time()
            structure_predictions = self.structure_models( #run trRosettaEnsemble -> runs trrosetta n times
                model_input, use_n_models=cfg.n_models
            )
            nntime += time.time() - nn_start


            #loss functions
            loss_start = time.time()
            E_curr, metrics = self.loss(
                seq_curr, structure_predictions, msa1hot, track=True
            )
            losstime += time.time() - loss_start


            #scoring evaluation
            score_start = time.time()
            if E_curr < self.best_E:
                self.best_bkg = metrics["background_loss"]
                self.best_mtf = metrics["motif_loss"]
                self.best_site = metrics["site_loss"]
                self.best_distogram_distribution = structure_predictions['dist']


            seq, E = self.metropolis(seq, seq_curr, E_curr, E)

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
                print(f"STATS      || loss: {E_curr:.3f}, bkg: {background_loss:.3f}, mtf: {mtf_loss:.3f}, site: {site_loss:.3f}")
                print(f"BEST STATS || loss: {self.best_E:.3f}, bkg: {b_bkg:.3f}, mtf: {b_mtf:.3f}, site: {b_site:.3f}")

                #timings
                total_time = time.time() - start_time
                print(f"total time: {total_time:.1f}")
                print(f"mut time:   {muttime:.2f}\t| {100*muttime/total_time:.2f}%")
                print(f"nn time:    {nntime:.1f}\t| {100*nntime/total_time:.2f}%")
                print(f"loss time:  {losstime:.2f}\t| {100*losstime/total_time:.2f}%")
                print(f"score time: {scoretime:.1f}\t| {100*scoretime/total_time:.2f}%")
                print(f"misc time:  {misctime:.2f}\t| {100*misctime/total_time:.2f}%")

                plot_muts(self.gradient, self.results_dir / "gradient.jpg")
                plot_muts(self.good_mutations, self.results_dir / "mutations.jpg")

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
                    print(f"\n--- Current best: {self.best_sequence}")

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
                    print("ending due to command")
                    break
                elif line == 'matrix':
                    if lines[1].strip() == 'n': cfg.MATRIX = False
                    else: cfg.MATRIX = True
                    print("MATRIX: " + str(cfg.MATRIX))
                elif line == 'gradient':
                    if lines[1].strip() == 'n': cfg.GRADIENT = False
                    else: cfg.GRADIENT = True
                elif line == "beta":
                    try:
                        value = float(lines[1].strip())
                        self.beta = value
                    except: print("input error")
                elif line == "bkw":
                    try:
                        value = float(lines[1].strip())
                        self.bkg_weight = value
                    except: print("input error")
                elif line == "aaw":
                    try:
                        value = float(lines[1].strip())
                        self.aa_weight = value
                    except: print("input error")
                elif line == "mw":
                    try:
                        value = float(lines[1].strip())
                        self.motif_weight = value
                    except: print("input error")
                elif line == "pause":
                    while True:
                        with open('control.txt', 'r') as reader:
                            line = reader.readlines()[0].strip()
                        if line != "pause":
                            break
                        else:
                            print("pause...")
                            time.sleep(30)

            if (delta_step_best > 300 and self.step > 1500 and self.step % 100 == 0) or (cfg.FAST and self.step > 499 and self.step % 50 == 0):
                std = np.std(np.array(E_tracker)[-1000:])
                n_std = std / np.array(E_tracker)[-1000:].mean()
                print("sd: " + str(std))
                if abs(std) < 0.01 or (cfg.FAST and abs(std) < 0.02):
                    break

            if self.step % self.M == 0 and self.step != 0:
                seq = self.fixup_MCMC(seq)

            misctime += time.time() - misc_start

        ########################################

        # Save final results before exiting:
        seq_curr = torch.from_numpy(
            aa2idx(self.best_sequence).reshape([1, self.seq_L])
        ).long()
        model_input, msa1hot = prep_seq(seq_curr)
        structure_predictions = self.structure_models(model_input)
        E_curr, self.best_metrics = self.loss(
            seq_curr, structure_predictions, msa1hot, track=True
        )

        #for key in self.best_metrics.keys():
        #    self.best_metrics[key] = v(metrics[key])

        TM_score_proxy = top_prob(structure_predictions['dist'], verbose=False)
        TM_score_proxy = TM_score_proxy[0]  # We're running with batch_size = 1

        self.best_metrics["TM_score_proxy"] = TM_score_proxy
        self.best_metrics["sequence"] = self.best_sequence
        self.best_metrics["motifweight"] = self.motif_weight
        self.best_metrics["motifmode"] = self.motifmode
        self.best_metrics["steps"] = self.step
        self.best_metrics["motifs"] = "\"" + str(self.motifs.copy()) + "\""

        # Dump distogram:
        best_distogram_distribution = structure_predictions['dist'].detach().cpu().numpy()
        distogram = distogram_distribution_to_distogram(best_distogram_distribution)
        plot_distogram(
            distogram,
            self.results_dir / f"result.jpg",
            clim=cfg.limits["dist"],
        )
        plot_progress(
            E_tracker,
            self.results_dir / "progress.jpg",
            title=f"Optimization curve after {self.step} steps",
        )

        # Write results to csv:
        with (self.results_dir / "result.csv").open("w") as f:
            for key, val in self.best_metrics.items():
                f.write(f"{key},{val}\n")

        with (self.results_dir / "mutation_log.csv").open("w") as f:
            for mut in self.mutation_log:
                f.write(f"{mut}\n")


        return self.best_metrics

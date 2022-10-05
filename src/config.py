"""trDesign configuration parameters."""

import numpy as np
from datetime import datetime

### Design Config ###
# Predefined start requires mmotifs and best_seq properties to be set
# Will pull LEN, sequence constraint from best_seq
# Will take dist restraints from motif file (.npz)
USE_PREDEFINED_START = False

# Set a random seed?
# np.random.seed(seed=1234)

LEN = 170  # sequence length
AA_WEIGHT = 1  # weight for the AA composition biasing loss term
BKG_WEIGHT = 1 # weight for background loss
RM_AA = "C"  # comma-separated list of specific amino acids to disable from being sampled (ex: 'C,F')
n_models = 5  # How many structure prediction models to ensemble? [1-5]
report_interval = 120 #seconds

TEMPLATE = False
TEMPLATE_MODE = 'motifs' #msa, motifs, predefined
USE_WEIGHTED_IDX = False #good, reciprocal, tm
OPTIMIZER = 'none' #none, gd, gd_pssm, msa, pssm, conprob, matrix, niter_X[_Y] (X = num of muts per iter, Y duration of X)
FILE_MATRIX = 'blosum62.txt' #blosum62, pepstruc, fft_290_nobkg
FILE_PSSM = None
FILE_MSA = None
DYNAMIC_MOTIF_PLACEMENT = True
PREDEFINED_MOTIFS = False

BACKGROUND = True

FORCECPU = False
FAST = False #lower threshold for simulation end parameters
TRACE = False #dump sequence + distogram at each improvement

if FORCECPU: #CPU is very slow, 256aa, 5 models is ~15 sec per mutation
    n_models = 1

# MCMC schedule:
MCMC = {}
MCMC["BETA_START"] = 25  # Energy multiplier for the metropolis criterion, higher value -> less likely to accept bad mutation
MCMC["N_STEPS"] = 5000  # Number of steps for each MCMC optimization
MCMC["COEF"] = 1.25 #1.25  # Divide BETA by COEF
MCMC["M"] = 100 #MCMC["N_STEPS"] // 200  # Adjust beta every M steps
MCMC["MAX"] = 3000
MCMC["BAD"] = 0.02
MCMC["T_LIMIT"] = 5.9 #hours
MCMC["T_START"] = datetime.now()

num_simulations = 1000  # Number of sequences to design

# seed_filepath = "trdesign-seeds.txt" # Optionally, start from a .txt file with sequences
seed_filepath =  None #'/home/frederik/Documents/inputseq.txt' # Sample starting sequences 100% at random

# Constraint can be specified as an .npz file containing ['dist', 'omega', 'theta', 'phi'] target arrays of shape LxL
# target_motif_path = 'target_motifs/target.npz'
target_motif_path = 'AP.npz'


use_motifs = True
use_sites = False #repulsive contraint (prob for shorter dist higher than prob for correct dist = massive penalty)
motif_placement_mode = 0 #0 = random position, 1 = dynamic, 2 = input order, 2.1 = input order even spread, 2.2 input order, no end overhang, 3 = order by group, 4 = order by dist, 5 = order by C->N dist,  -1 = random mode
use_random_length = False #uses random protein length between length of motifs and the specified LEN
use_random_motif_weight = False
motif_weight_max = 10 #min weight is 1
first_residue_met = True
PSSM = None

# Restraint map generated from pymol script
sequence_constraint = '''NRAAQGDITAPGGARRLTGDQTAALRDSLSDKPAKNIILLIGDGMGDSEITAARNYAEGAGGFFKGIDALPLTGQYTHYALNKKTGKPDYVTDSAASATAWSTGVKTYNGALGVDIHEKDHPTILEMAKAAGLATGNVSTAELQDATPAALVAHVTSRKCYGPSATSEKCPGNALEKGGKGSITEQLLNARADVTLGGGAKTFAETATAGEWQGKTLREQAQARGYQLVSDAASLNSVTEANQQKPLLGLFADGNMPVRWLGPKATYHGNIDKPAVTCTPNPQRNDSVPTLAQMTDKAIELLSKNEKGFFLQVEGASIDKQDHAANPCGQIGETVDLDEAVQRALEFAKKEGNTLVIVTADHAHASQIVAPDTKAPGLTQALNTKDGAVMVMSYGNSEEDSQEHTGSQLRIAAYGPHAANVVGLTDQTDLFYTMKAALGL'''.replace('\n','')
motif_constraint =    '''------------------------------------------mm------------------------------------------------mmm-------------------------------------------------mmm----------m-----------------------------------------------------------------------------------------------------------------------------------------------------------m-m--mm--m-------------------------------------mm-m---------------------------------------m------------------------------------'''.replace('\n','')
motif_position =      '''------------------------------------------11------------------------------------------------111-------------------------------------------------111----------1-----------------------------------------------------------------------------------------------------------------------------------------------------------1111111111-------------------------------------1111---------------------------------------1------------------------------------'''.replace('\n','')

#predefined start motifs (list of motifs [[[m1],[m2],[m3]],[[m1],[m2],[m3]]])
mmotifs = [
[[34, 54, 21, '--rrcccrmrr----------', 1, 0, 20], [71, 76, 6, 'rrrrrr', 1, 22, 27], [83, 102, 20, '---------gmccccrrr--', 1, 28, 47], [121, 181, 61, '--rrrrrrrrrrrcccccc--rrgcmccccrrrrrrmr-----------------------', 1, 58, 118], [286, 366, 81, '-----------------------ccccmgcccmgccmccccccccrrrrrrrrrrrrrrrrrrrrrrrrrccccmmr----', 1, 125, 205], [397, 437, 41, '-----rmrrrrrrrrr-------------rrrrrrrrrrrr', 1, 221, 261]]
]

best_seq = "MAVILVFIDMGSFKRYKQRYPEFYKQARKRGMSYYGHDSWGLGWALAEFVLELLKKSNLTVIPDYQEKGKPVIIVDPGKGGDWTHMILPRVFGHRTDSSPWHQKYHAKLRKVLQQKGIKPLNVRRYNDDDTPEQRAKRLIELAKSGPVVILIEGHAYDKYWHNGDKKKSDRHMEEAVEVLRAVAEAISKEKQVYTLIIGDHGFSYSPRQMKSLVDNGHVKLLIFNRSHHPDMWMVKLVSPDARIVDTPEVFARFVELAIRRL"

sequence_restraint_letters = "mg"
structure_restraint_letters = "mygcr"
structure_restraint_mask_values = {'m': 12, 'y': 12, 'g': 5, 'c': 5, 'r': 2} 


#Setup predefined start 
if USE_PREDEFINED_START:
    TEMPLATE = True
    TEMPLATE_MODE = 'predefined' #Can also be msa
    PREDEFINED_MOTIFS = True
    LEN = len(best_seq)
    MCMC["BETA_START"] = 200

###################################################
### Below are thinks that should not be touched ###
###################################################

#MISC setup
varlen = ""
if use_random_length: varlen = "var"
experiment_name = datetime.now().strftime("%Y-%m-%d") + f"_{varlen}{LEN}aa"

### Constants  ###
# These settings are specific to the trRosetta Model implementation
# (might need to change for AF2)
limits = {
    "dist": [2, 20],
    "omega": [-np.pi, np.pi],
    "theta": [-np.pi, np.pi],
    "phi": [0, np.pi],
}

bin_dict_np = {
    # Inter-residue distances (Angstroms)
    "dist": np.linspace(*limits["dist"], num=37)[:-1] + 0.25,
    # Omega-angles (radians)
    "omega": np.linspace(*limits["omega"], num=25)[:-1] + np.pi / 24,
    # Theta-angles (radians)
    "theta": np.linspace(*limits["theta"], num=25)[:-1] + np.pi / 24,
    # Phi-angles (radians)
    "phi": np.linspace(*limits["phi"], num=13)[:-1] + np.pi / 24,
}

# Add "no-contact" values:
no_contact_value = np.inf
for key in bin_dict_np:
    bin_dict_np[key] = np.insert(bin_dict_np[key], 0, no_contact_value)

print_dist_bins = 0
if print_dist_bins:
    for i, midpoint in enumerate(bin_dict_np["dist"]):
        left = midpoint - 0.25
        right = midpoint + 0.25
        print(f"Bin {i:02}: [{left:.2f} - {right:.2f}] ({midpoint:.2f})")

### Amino Acid Alphabet ###

ALPHABET_core_str = "ARNDCQEGHILKMFPSTWYV"  # exclude "-" gap char

ALPHABET_full_str = ALPHABET_core_str + "-"
MAX_AA_INDEX = len(ALPHABET_full_str) - 1

ALPHABET_core = np.array(list(ALPHABET_core_str), dtype="|S1").view(np.uint8)  # no "-"
ALPHABET_full = np.array(list(ALPHABET_full_str), dtype="|S1").view(np.uint8)

AA_PROPERTY_GROUPS = {
    'proline'       : 'P',
    'negative'      : 'ED',
    'positive'      : 'HKR',
    'tiny'          : 'AGS',
    'aliphatic'     : 'ILV',
    'aromatic'      : 'FYWH',
    'charged'       : 'HKRED',
    'helix'         : 'ARMLK',
    'small'         : 'VCAGDNSTP',
    'polar'         : 'YWHKREQDNST',
    'large'         : 'WYFRMILQEHKR',
    'hydrophobic'   : 'ILVCAGMFYWHKT',
}

AA_GROUPS = {
    'negative'      : 'ED',
    'positive'      : 'HKR',
    'leu'           : 'L',
    'ile'           : 'I',
    'isol'          : 'IL',
    'val'           : 'V',
    'ala'           : 'A',
    'arg'           : 'R',
    'asn'           : 'N',
    'asp'           : 'D',
    'aspn'          : 'ND',
    'cys'           : 'C',
    'gln'           : 'Q',
    'glu'           : 'E',
    'glun'          : 'GE',
    'gly'           : 'G',
    'his'           : 'H',
    'lys'           : 'K',
    'met'           : 'M',
    'phe'           : 'F',
    'pro'           : 'P',
    'ser'           : 'S',
    'thr'           : 'T',
    'sert'          : 'ST',
    'trp'           : 'W',
    'tyr'           : 'Y',
}

### Target Amino Acid Distribution ###


# ALPHABET_core_str = "ARNDCQEGHILKMFPSTWYV"
# Using all of PDB:
# fmt: off
native_freq = np.array([0.078926, 0.049790, 0.045148, 0.060338, 0.012613,
                        0.037838, 0.065925, 0.071221, 0.023248, 0.056478,
                        0.093113, 0.059803, 0.020729, 0.041453, 0.046319,
                        0.061237, 0.054742, 0.014891, 0.037052, 0.069127])

# Using PDB filtered to [40-100] residue range (total of 47,693 sequences):
print(" ---- Using PDB [40-100] native frequencies! ----")
native_freq = np.array([0.075905, 0.070035, 0.039181, 0.045862, 0.023332,
                        0.035662, 0.066048, 0.064150, 0.021644, 0.059121,
                        0.089042, 0.084882, 0.031276, 0.035995, 0.038211,
                        0.060108, 0.053137, 0.008422, 0.026804, 0.071172])
# fmt: on

motif_placement_mode_dict = {
    0   : 'random placement',
    1   : 'dynamic',
    2   : 'input order',
    2.1 : 'input order even spread',
    2.2 : 'input order, no end overhang, SD spacing',
    2.3 : 'input order, no end overhang, random spacing',
    3   : 'order by group',
    3.2 : 'order by group, no end overhang, SD spacing',
    3.3 : 'order by group, no end overhang, random spacing',
    4   : 'order by dist',
    5   : 'order by C->N dist',
    6   : 'rotate order',
    6.2 : 'rotate order, no end overhang, SD spacing',
    6.3 : 'rotate order, no end overhang, random spacing',
    -1  : 'random mode',
    -3  : 'predefined'
}
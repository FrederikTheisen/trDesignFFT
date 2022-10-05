"""trDesign configuration parameters."""

import numpy as np
from datetime import datetime

### Design Config ###
# Predefined start requires mmotifs and best_seq properties to be set
# Will pull LEN, sequence constraint from best_seq
# Will take dist restraints from motif file (.npz)
USE_PREDEFINED_START = False

LEN = 170  # sequence length
USE_RANDOM_LENGTH = False  #uses random protein length between length of motifs and the specified LEN
RM_AA = "C"  # comma-separated list of specific amino acids to disable from being sampled (ex: 'C,F')
FIRST_RESIDUE_MET = True
n_models = 5  # How many structure prediction models to ensemble? [1-5]
report_interval = 120 #seconds

TEMPLATE = False
TEMPLATE_MODE = 'motifs' #msa, msa_argmax, motifs, predefined
FILE_MSA = None

USE_WEIGHTED_IDX = False #good, reciprocal, tm

### OPTIMIZER ###
#Selects mutations based on various properties.
#msa and conprob are MSA based. msa is based on jalview residue properties. conprob is based on probabilities
#pssm reads a an msa with scores for each position and each amino acid. mutations are selected based on score
#matrix uses substitution matrix and will decide mutation based on previous delta score
#can include _niter_X_Y which will do X mutations for Y steps and then 1 mutation every step 
#can include '_start' which will change the mode to 'none' after 4xLEN steps
OPTIMIZER = 'none' #MUST be string. Options: none, gd, gd_pssm, msa, pssm, conprob, matrix, _niter_X[_Y], _start
FILE_MATRIX = 'blosum62.txt' #Options: blosum62, pepstruc, fft_290_nobkg
FILE_PSSM = None

DYNAMIC_MOTIF_PLACEMENT = True
PREDEFINED_MOTIFS = False

### LOSS FUNCTIONS ###
BACKGROUND = True #Background loss function, designs folded protein
BKG_WEIGHT = 1 # weight for background loss
MOTIFS = True #Motif loss, design target motif
MTF_WEIGHT = 1
SITE = False #Designs pocket defined by m restraints, repulsive loss function
AA_WEIGHT = 1  # weight for the AA composition biasing loss term

### MCMC schedule ###
MCMC = {}
MCMC["BETA_START"] = 25  # Energy multiplier for the metropolis criterion, higher value -> less likely to accept bad mutation
MCMC["N_STEPS"] = 5000  # Number of steps for each MCMC optimization
MCMC["COEF"] = 1.25 #1.25  # Divide BETA by COEF
MCMC["M"] = 100 #MCMC["N_STEPS"] // 200  # Adjust beta every M steps
MCMC["MAX"] = 3000 #Maximum BETA value
MCMC["BAD"] = 0.02 #Minimum fraction of bad mutation accepted. If fewer are accepted, then BETA is descreased (default 0.05)
MCMC["T_LIMIT"] = 5.9 #Time limit for supercomputer computation scheduling. 

# Constraint can be specified as an .npz file containing ['dist', 'omega', 'theta', 'phi'] target arrays of shape LxL
# target_motif_path = 'target_motifs/target.npz'
target_motif_path = 'AP.npz'



motif_placement_mode = 0 #0 = random position, 1 = dynamic, 2 = input order, 2.1 = input order even spread, 2.2 input order, no end overhang, 3 = order by group, 4 = order by dist, 5 = order by C->N dist,  -1 = random mode
use_random_motif_weight = False


# Restraint map generated from pymol script
sequence_constraint = '''NRAAQGDITAPGGARRLTGDQTAALRDSLSDKPAKNIILLIGDGMGDSEITAARNYAEGAGGFFKGIDALPLTGQYTHYALNKKTGKPDYVTDSAASATAWSTGVKTYNGALGVDIHEKDHPTILEMAKAAGLATGNVSTAELQDATPAALVAHVTSRKCYGPSATSEKCPGNALEKGGKGSITEQLLNARADVTLGGGAKTFAETATAGEWQGKTLREQAQARGYQLVSDAASLNSVTEANQQKPLLGLFADGNMPVRWLGPKATYHGNIDKPAVTCTPNPQRNDSVPTLAQMTDKAIELLSKNEKGFFLQVEGASIDKQDHAANPCGQIGETVDLDEAVQRALEFAKKEGNTLVIVTADHAHASQIVAPDTKAPGLTQALNTKDGAVMVMSYGNSEEDSQEHTGSQLRIAAYGPHAANVVGLTDQTDLFYTMKAALGL'''.replace('\n','')
motif_constraint =    '''------------------------------------------mm------------------------------------------------mmm-------------------------------------------------mmm----------m-----------------------------------------------------------------------------------------------------------------------------------------------------------m-m--mm--m-------------------------------------mm-m---------------------------------------m------------------------------------'''.replace('\n','')
motif_position =      '''------------------------------------------11------------------------------------------------111-------------------------------------------------111----------1-----------------------------------------------------------------------------------------------------------------------------------------------------------1111111111-------------------------------------1111---------------------------------------1------------------------------------'''.replace('\n','')


#predefined start motifs (list of motif combinations [[[m1],[m2],[m3]],[[m1],[m2],[m3]]])
mmotifs = [
[[34, 54, 21, '--rrcccrmrr----------', 1, 0, 20], [71, 76, 6, 'rrrrrr', 1, 22, 27], [83, 102, 20, '---------gmccccrrr--', 1, 28, 47], [121, 181, 61, '--rrrrrrrrrrrcccccc--rrgcmccccrrrrrrmr-----------------------', 1, 58, 118], [286, 366, 81, '-----------------------ccccmgcccmgccmccccccccrrrrrrrrrrrrrrrrrrrrrrrrrccccmmr----', 1, 125, 205], [397, 437, 41, '-----rmrrrrrrrrr-------------rrrrrrrrrrrr', 1, 221, 261]]
]

best_seq = "MAVILVFIDMGSFKRYKQRYPEFYKQARKRGMSYYGHDSWGLGWALAEFVLELLKKSNLTVIPDYQEKGKPVIIVDPGKGGDWTHMILPRVFGHRTDSSPWHQKYHAKLRKVLQQKGIKPLNVRRYNDDDTPEQRAKRLIELAKSGPVVILIEGHAYDKYWHNGDKKKSDRHMEEAVEVLRAVAEAISKEKQVYTLIIGDHGFSYSPRQMKSLVDNGHVKLLIFNRSHHPDMWMVKLVSPDARIVDTPEVFARFVELAIRRL"

sequence_restraint_letters = "mgs"
structure_restraint_letters = "mygcr"
structure_restraint_mask_values = {'m': 12, 'y': 12, 'g': 5, 'c': 5, 'r': 2} 


#Setup predefined start settings
if USE_PREDEFINED_START:
    TEMPLATE = True
    TEMPLATE_MODE = 'predefined' #Can also be msa
    PREDEFINED_MOTIFS = True
    LEN = len(best_seq)
    MCMC["BETA_START"] = 200 

###################################################
### Below are things that should not be touched ###
###################################################

# seed_filepath = "trdesign-seeds.txt" # Optionally, start from a .txt file with sequences
seed_filepath =  None #'/home/frederik/Documents/inputseq.txt' # Sample starting sequences 100% at random
num_simulations = 1000  # Number of sequences to design

#MISC setup
varlen = ""
if use_random_length: varlen = "var"
experiment_name = datetime.now().strftime("%Y-%m-%d") + f"_{varlen}{LEN}aa"

use_random_length = USE_RANDOM_LENGTH
motif_weight_max = MTF_WEIGHT
first_residue_met = FIRST_RESIDUE_MET
PSSM = None
MCMC["T_START"] = datetime.now()
use_motifs = MOTIFS
use_sites = SITE #repulsive contraint (prob for shorter dist higher than prob for correct dist = massive penalty)

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
"""trDesign configuration parameters."""

import numpy as np
from datetime import datetime

### Design Config ###

# Set a random seed?
# np.random.seed(seed=1234)

LEN = 256  # sequence length
AA_WEIGHT = 1  # weight for the AA composition biasing loss term
BKG_WEIGHT = 1 # weight for background loss
RM_AA = "C"  # comma-separated list of specific amino acids to disable from being sampled (ex: 'C,F')
n_models = 100  # How many structure prediction models to ensemble? [1-5]
FORCECPU = False

if FORCECPU: #CPU is very slow, 256aa, 100 models is ~15 sec per mutation
    n_models = 1

# MCMC schedule:
MCMC = {}
MCMC["BETA_START"] = 25  # Energy multiplier for the metropolis criterion, higher value -> less likely to accept bad mutation
MCMC["N_STEPS"] = 4000  # Number of steps for each MCMC optimization
MCMC["COEF"] = 1.1  # Divide BETA by COEF
MCMC["M"] = MCMC["N_STEPS"] // 200  # Adjust beta every M steps

num_simulations = 100  # Number of sequences to design

# seed_filepath = "trdesign-seeds.txt" # Optionally, start from a .txt file with sequences
seed_filepath =  None #'/home/frederik/Documents/inputseq.txt' # Sample starting sequences 100% at random

# Constraint can be specified as an .npz file containing ['dist', 'omega', 'theta', 'phi'] target arrays of shape LxL
# target_motif_path   = 'target_motifs/target.npz'
target_motif_path = '/home/frederik/Documents/enzyme.npz'


use_motifs = True
motif_placement_mode = 3 #0 = random position, 1 = dynamic, 2 = input order, 2.5 = input order even spread, 3 = order by group, 4 = order by dist, 5 = order by C->N dist,  -1 = random mode
use_random_length = False #uses random protein length between length of motifs and the specified LEN
motif_weight_max = 1 #min weight is 1

# keep certain positions at specific residues (e.g., "---A---C---")
sequence_constraint = '''MEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQNFR
FEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEITLTGKMLT'''.replace('\n','')
motif_constraint = '''--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ssssssssssssssssss----ssssssssssssssssssssssssssssbsssssssssssssssssssssssssssssssssssssssssssbsssssss---------------------sbbbss--------------sssssssssssssssssssssbbssbsssssssssssssssssssssssssssbssssssssssssbssssssbs-------------------------------------------sssssssbssssssssssssssssssssssss-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------b------------------------------------------------------------------------------------------------bbsb------------------------------------------------------
---------------------------------------------------------------------------------------------------------------'''.replace('\n','')
motif_position =   '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------1111111111111111111111----444444444444444444444444444444444444444444444444444444444444444444444444444444444-------------------55555555------66666666666666666666666666666666666666666666666666666666666666666666666666666666666-----------------------------------------777777777777777777777777777777777777777----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------222----------------------------------------------------------------------------------------------333333-----------------------------------------------------
---------------------------------------------------------------------------------------------------------------'''.replace('\n','')

#predefined start parameters
use_predef_start = False
motifs = [[550, 554, 5, 'sbbss', 0, 15, 19],
[298, 304, 7, 'ssbsbss', 0, 29, 35],
[839, 842, 4, 'sbss', 0, 38, 41],
[936, 943, 8, 'sbbssss-', 0, 50, 57],
[334, 346, 13, 'bss--ssbbssbs', 0, 59, 71],
[378, 385, 8, 'bsbssbsb', 0, 73, 80],
[406, 417, 12, 'ssbbbbssssss', 0, 96, 107],
[446, 455, 10, 'bsbbssbsbs', 0, 119, 128],
[478, 506, 29, '-sbbsss-------sbbsssssbss--s-', 0, 135, 163]]
best_seq = "KRKIFIVVQFPADTPFWQLRIMKELARRGVYMVFGSDFTKLTRYLRQAGIKEDFTFMRQKTTGKSSHDIGHGVHSNKPEKYDVVIFIGSMTEKEMTRSWNWLEESEPTEGDIIIHLSGGYGNVIPNSQLKEKGAIVVEWDDRMAKQGAQVHWLEPNSYSEMVRNLLSALD"
b_seq_cn = "----------------WQ-------------M-F-----K-----------ED------K------HD--H--H-N--E-Y-----------------WNWL-----------------Y-NV--N-Q---------EW-----------HW-----Y------------"

varlen = ""
if use_random_length: varlen = "var"
experiment_name = f"{varlen}{LEN}aa_{MCMC['N_STEPS']}steps_" + datetime.now().strftime("%Y-%m-%d_%H%M")


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

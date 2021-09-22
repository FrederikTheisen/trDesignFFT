"""trDesign configuration parameters."""

import numpy as np
from datetime import datetime

### Design Config ###

# Set a random seed?
# np.random.seed(seed=1234)

LEN = 290  # sequence length
AA_WEIGHT = 1  # weight for the AA composition biasing loss term
BKG_WEIGHT = 1 # weight for background loss
RM_AA = "C"  # comma-separated list of specific amino acids to disable from being sampled (ex: 'C,F')
n_models = 5  # How many structure prediction models to ensemble? [1-5]
report_interval = 60 #seconds

TEMPLATE = False
TEMPLATE_MODE = 'msa' #msa, motifs, predefined
GRADIENT = False
MATRIX = True
MATRIX_MODE = 'probability' #probability, msa, groups
MATRIX_DYNAMIC = False
MATRIXFILE = 'pepstruc.txt' #blosum62, pepstruc, fft_290_nobkg
MSA_FILE = 'msa_290.txt'

BACKGROUND = (BKG_WEIGHT > 0)

FORCECPU = False
FAST = False #lower threshold for simulation end parameters
TRACE = False #dump sequence + distogram at each improvement

if FORCECPU: #CPU is very slow, 256aa, 5 models is ~15 sec per mutation
    n_models = 1

# MCMC schedule:
MCMC = {}
MCMC["BETA_START"] = 300  # Energy multiplier for the metropolis criterion, higher value -> less likely to accept bad mutation
MCMC["N_STEPS"] = 20000  # Number of steps for each MCMC optimization
MCMC["COEF"] = 1.15 #1.25  # Divide BETA by COEF
MCMC["M"] = 100 #MCMC["N_STEPS"] // 200  # Adjust beta every M steps
MCMC["MAX"] = 5000

num_simulations = 200  # Number of sequences to design

# seed_filepath = "trdesign-seeds.txt" # Optionally, start from a .txt file with sequences
seed_filepath =  None #'/home/frederik/Documents/inputseq.txt' # Sample starting sequences 100% at random

# Constraint can be specified as an .npz file containing ['dist', 'omega', 'theta', 'phi'] target arrays of shape LxL
# target_motif_path   = 'target_motifs/target.npz'
target_motif_path = '/home/frederik/Documents/EngBF_unbound.npz'


use_motifs = True
use_sites = False
motif_placement_mode = 2.2 #0 = random position, 1 = dynamic, 2 = input order, 2.1 = input order even spread, 2.2 input order, no end overhang, 3 = order by group, 4 = order by dist, 5 = order by C->N dist,  -1 = random mode
use_random_length = False #uses random protein length between length of motifs and the specified LEN
use_random_motif_weight = False
motif_weight_max = 1 #min weight is 1
first_residue_met = True

# keep certain positions at specific residues (e.g., "---A---C---")
#290 residue motif
sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
motif_constraint = '''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rrrrrrrrrrrrcgcgcgcrrrrrrrrrrrrrrrrrrrrrrrrrrrcgcgcrrrrcggccmcrrrrrrrrrrrrrrrrrrrrrrrrrrrrcccmcmrrgrrrrrrrrrrrrrrrrrrrrrrrcgggcccccrrlrrrrrrrrrrrrrrrrrrrrrrrrrrcgcmgrrgrgrrlrrrrrrrrrrrrrrrrrrrcccmgrrrrrrrrrgcggrrcrcgrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrcggcrrrrrrrrrrrrrrrrrrr------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')
motif_position =   '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')

sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
motif_constraint = '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rccccccccgccgcgcgcgcrrrrrrrrrrrrrrrrrrrrrrrrrcgcgcmcrrrrcmgccmcrrrrrrrrrrrrrrrrrrrrrrrrrrcgcgcmcmccgcrrrrrrrrrrrrr-------rrcmcmccccrrrrrrrrrrrrrrrrrrrrrrrrrrrrrgcggmgccgcgrrrrrrrrrrrrrrrrrrrrrccggmggggrrrrrrgcgmccccgmgccrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrcmcccmggcrrrrrrrrrrrrrrrrrrrrrrr-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')
motif_position =   '''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-----111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')


#256 residue motif
#sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
#FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
#motif_constraint = '''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rrrrrrrrrrccggcgcgc------rrrrrrrrrrrrrrrrrrrcccggggrrrrcggccmcrr----------rrrrrrrrrrrrrrcccggmgmgggcrrrrrr----------------cmgmcgcccr-----------rrrrrrrrrr-----rccgcmggggcgrrrrrrrrrrrrrrrrrrrrrccccmgrr------rgcggccgccgccc----------------------------------------rrccccgmggcrrrrrrrrrr--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------'''.replace('\n','')
#motif_position =   '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111----------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-----------------111111111111111111111111111111------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------'''.replace('\n','')



#predefined start parameters
use_predef_motif = False
motifs = [[287, 353, 67, '-rrrrrrrrrrrrgrgrgr------rrrrrrrrrrrrrrrrrrrrrrgrgrrrrrrggrrmrrr---', 1, 0, 66],
[358, 393, 36, '---rrrrrrrrrrrrrrrrrrrmrmrrgrr------', 1, 69, 104],
[405, 517, 113, '----rgggr------rrrrrrrrrrrrrrrrrrrrrrrrrrgrgrmgrrgrgrrrrrrrrrrrrrrrrrrrrgggrgmgrrrrrrrrrgrggrrrrrgrrrrrrr--------', 1, 106, 218],
[539, 569, 31, '-------rgrgrrrggrrrrrrrrrrrr---', 1, 225, 255]]

use_predef_start = False

#290 residue start seq
best_seq = "MLMLYKHQWTIILYVIVMTFGATPMATITEVEAFLEKMAELGKGIVVGVLVKFHAFGVHDFHHFHLGKALAAGKFKKTVKALATKMKANFVHIGIHLNIFEMGHTEAKKTGKLPKSMAGKWNWGDTWKYDAVKKPTVKKMKKWIKLLKAQKFVGLAFLYVDVFYNTQLALGTKKTYTKLHKTLKAHGVIYAIEWGHGFAYKHVFVHWWGFNTYGLGAAKKLDGMKTKDGGKIAMIVGGVKTKGTEKKAGRKYKVTAVLLDVGIWQGFFMDKKLVTALLKYIKTAKKHLTW"
#best_seq = "MQIKKGKEVVILSRIIMYFGPSPEQFAEQVEKALEALEKLPEKYIIMVLLKGFGGGGHDTFHYKQIKEEQMERAIEAAQALKKQGLDKTVMVGIHVNASEPSPDYQQDPEMKKRWNWGDKPDKEEMREYHRRFMRRAMRALQKRGFKTMVHIYSDVWGNEQLPGQSEEERQEMRKEFKQTGIYVHTEWNMMEGADHTFVHWYWDTSYASEADAKRVIIIGNSSHMERLLKNGKVLLAPFGWQGFALENAKRVLYLP"

varlen = ""
if use_random_length: varlen = "var"
varstart = ""
if use_predef_motif: varstart = "pm"
if use_predef_start: varstart += "ps"
experiment_name = datetime.now().strftime("%Y-%m-%d") + f"_{varlen}{LEN}aa"

if use_predef_motif or use_predef_start: experiment_name += "_" + varstart

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
sequence_restraint_letters = "gyml"
structure_restraint_letters = "rgmcl"#"rgmcl"
structure_restraint_mask_values = { "r" : 1, "g" : 4, "m" : 30 , "c" : 2, "l" : 2 }#{ "r" : 1, "g" : 6, "m" : 11 , "c" : 5, "l" : 2 }

motif_placement_mode_dict = {0:'random placement',1:'dynamic',2:'input order', 2.1 : 'input order even spread', 2.2 :'input order, no end overhang', 3 : 'order by group', 4 : 'order by dist', 5 : 'order by C->N dist',  -1 : 'random mode'}

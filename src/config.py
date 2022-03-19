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
USE_WEIGHTED_IDX = False #good, reciprocal, tm
OPTIMIZER = 'none' #none, gd, gd_pssm, msa, pssm, conprob, matrix, niter_X[_Y] (X = num of muts per iter, Y duration of X)
FILE_MATRIX = 'matrix_fft2.txt' #blosum62, pepstruc, fft_290_nobkg
FILE_PSSM = 'pssm_290.txt'
FILE_MSA = 'msa_532.txt'
DYNAMIC_MOTIF_PLACEMENT = False
PREDEFINED_MOTIFS = False

BACKGROUND = True

FORCECPU = False
FAST = False #lower threshold for simulation end parameters
TRACE = False #dump sequence + distogram at each improvement

if FORCECPU: #CPU is very slow, 256aa, 5 models is ~15 sec per mutation
    n_models = 1

# MCMC schedule:
MCMC = {}
MCMC["BETA_START"] = 50  # Energy multiplier for the metropolis criterion, higher value -> less likely to accept bad mutation
MCMC["N_STEPS"] = 2500  # Number of steps for each MCMC optimization
MCMC["COEF"] = 1.25 #1.25  # Divide BETA by COEF
MCMC["M"] = 100 #MCMC["N_STEPS"] // 200  # Adjust beta every M steps
MCMC["MAX"] = 3000
MCMC["BAD"] = 0.02

num_simulations = 1000  # Number of sequences to design

# seed_filepath = "trdesign-seeds.txt" # Optionally, start from a .txt file with sequences
seed_filepath =  None #'/home/frederik/Documents/inputseq.txt' # Sample starting sequences 100% at random

# Constraint can be specified as an .npz file containing ['dist', 'omega', 'theta', 'phi'] target arrays of shape LxL
# target_motif_path = 'target_motifs/target.npz'
target_motif_path = '/home/frederik/Documents/EngBF_unbound.npz'


use_motifs = True
use_sites = False
motif_placement_mode = 2.3 #0 = random position, 1 = dynamic, 2 = input order, 2.1 = input order even spread, 2.2 input order, no end overhang, 3 = order by group, 4 = order by dist, 5 = order by C->N dist,  -1 = random mode
use_random_length = False #uses random protein length between length of motifs and the specified LEN
use_random_motif_weight = False
motif_weight_max = 1 #min weight is 1
first_residue_met = True
PSSM = None

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

sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
motif_constraint = '''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rcmcmc----------------------------rcmc----cmmcr-------------------------------rcmcmcr-----------------------rcgcg---------------------------------rcmcmcccgcgcr---------------------rcmc---------rcmgcrrrcgcr-------------------------------------------rmcccmgr----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------g-------------------------------------------------------------------------------------------------m------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')
motif_position =   '''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------111111111-------------------------1111111----11111----------------------------1111111111-----------------------11111---------------------------------1111111111111---------------------1111---------111111111111-------------------------------------------11111111----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------1-------------------------------------------------------------------------------------------------1------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')

sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
motif_constraint = '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rrrrrrcrcccccccm---------------------rrrrrcccccmcrrrrcmmccmcrr----------------------rrcccccmcmccmc----------------------cgggcccccrrrrrrrrrrr--------------rrrccmgmmccgcmrrrrr-------------rrrccccmgccc------ccmmcccccmccc------------------c--------------------ccccmcccmmccrrrrrrr--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------m------------------------------------------------------------------------------------------------gm------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')
motif_position =   '''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------11111111111111111---------------------4444444444444444444444444----------------------44444444444444444444-----------4444444444444444444444444--------------4444444444444444444-------------444444444444444-444444444444444----------------4444444----------------44444444444444444444444444444444444--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------222------------------------------------------------------------------------------------------------3333----------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')

#restraints for 17/3/2022 start
sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
motif_constraint = '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rrrrrrcrcccccccgrrrrrrrrrrrrrrrrrrrrrrrrrrcccccmcrrrrcmmccmcrrrrrrrrrrrrrrrrrrrrrrrrrrcccccmcmccgcrrrrr----------------rcgggcccccrrrrrrrrrrrrrrrrrrrrrrrrrrrrccggmmccgcgrrrrrrrrrrrrrrrrrrrrrccccmgccc------ccggcccccgcccrrrrrrrrrrrrrrrrrrcrrr-----------------ccccccccggccrrrrrrrrrrrrrrrrrrrrrrr-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')
motif_position =   '''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111------111111111111111111111111111111111111111111111111111111111111111111111111111111111111-11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------'''.replace('\n','')


#256 residue motif
#sequence_constraint = '''SHMEKETGPEVDDSKVTYDTIQSKVLKAVIDQAFPRVKEYSLNGHTLPGQVQQFNQVFINNHRITPEVTYKKINETTAEYLMKLRDDAHLINAEMTVRLQVVDNQLHFDVTKIVNHNQVTPGQKIDDESKLLSSISFLGNALVSVSSDQTGAKFDGATMSNNTHVSGDDHIDVTNPMKDLAKGYMYGFVSTDKLAAGVWSNSQNSYGGGSNDWTRLTAYKETVGNANYVGIHSSEWQWEKAYKGIVFPEYTKELPSAKVVITEDANADKNVDWQDGAIAYRSIMNNPQGWEKVKDITAYRIAMNFGSQAQNPFLMTLDGIKKINLHTDGLGQGVLLKGYGSEGHDSGHLNYADIGKRIGGVEDFKTLIEKAKKYGAHLGIHVNASETYPESKYFNEKILRKNPDGSYSYGWNWLDQGINIDAAYDLAHGRLARWEDLKKKLGDGLDFIYVDVWGNGQSGDNGAWATHVLAKEINKQGWRFAIEWGHGGEYDSTFHHWAADLTYGGYTNKGINSAITRFIRNHQKDAWVGDYRSYGGAANYPLLGGYSMKDFEGWQGRSDYNGYVTNLFAHDVMTKYFQHFTVSKWENGTPVTMTDNGSTYKWTPEMRVELVDADNNKVVVTRKSNDVNSPQYRERTVTLNGRVIQDGSAYLTPWNWDANGKKLSTDKEKMYYFNTQAGATTWTLPSDWAKSKVYLYKLTDQGKTEEQELTVKDGKITLDLLANQPYVLYRSKQTNPEMSWSEGMHIYDQGFNSGTLKHWTISGDASKAEIVKSQGANDMLRIQGNKEKVSLTQKLTGLKPNTKYAVYVGVDNRSNAKASITVNTGEKEVTTYTNKSLALNYVKAYAHNTRRNNATVDDTSYFQNMYAFFTTGADVSNVTLTLSREAGDEATYFDEIRTFENNSSMYGDKHDTGKGTFKQDFENVAQGIFPFVVGGVEGVEDNRTHLSEKHDPYTQRGWNGKKVDDVIEGNWSLKTNGLVSRRNLVYQTIPQN
#FRFEAGKTYRVTFEYEAGSDNTYAFVVGKGEFQSQASNLEMHELPNTWTDSKKAKKATFLVTGAETGDTWVGIYSTGNASNTRGDSGGNANFRGYNDFMMDNLQIEEI'''.replace('\n','')
#motif_constraint = '''-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------rrrrrrrrrrccggcgcgc------rrrrrrrrrrrrrrrrrrrcccggggrrrrcggccmcrr----------rrrrrrrrrrrrrrcccggmgmgggcrrrrrr----------------cmgmcgcccr-----------rrrrrrrrrr-----rccgcmggggcgrrrrrrrrrrrrrrrrrrrrrccccmgrr------rgcggccgccgccc----------------------------------------rrccccgmggcrrrrrrrrrr--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------'''.replace('\n','')
#motif_position =   '''----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111----------11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111-----------------111111111111111111111111111111------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------'''.replace('\n','')



#predefined start parameters
use_predef_motif = PREDEFINED_MOTIFS
motifs = [[288, 389, 102, '-occcccoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgcg--', 1, 0, 101], [406, 426, 21, '---gmcmcccccoo-------', 1, 106, 124], [428, 489, 62, '-----------------occgcmgccgcgo--------------------occcmgccc---', 1, 132, 193], [491, 520, 30, '--ccmgcccccgcccooor-----------', 1, 199, 228], [531, 579, 49, '----------------roooccmgco-----------------------', 1, 241, 289]]

#mmotifs = [[[289, 397, 109, 'oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 1, 109], [406, 486, 81, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc', 1, 120, 200], [493, 526, 34, 'ccmgcccccgcccocor-----------roocor', 1, 217, 250], [543, 556, 14, 'yoccccccccmgcc', 1, 276, 289]]]
mmotifs = [
[[532, 579, 48, '-------------rrrcccccmgccrrrrrrrrrrrrrr---------', 0, 2, 49], [285, 306, 22, '---rrrrrrrrrcccccgcgcr', 0, 51, 72], [313, 352, 40, '-------------rrrrrcccccmcrrrrcmgccmcrrr-', 0, 76, 115], [360, 389, 30, '-------rrrrrrrrcccccmcmccccr--', 0, 119, 148], [409, 427, 19, '-mcmrrrrrrrrrrr----', 0, 149, 167], [433, 459, 27, '------rrrrrrrccggmccccccrrr', 0, 180, 206], [465, 488, 24, '---------rrrrcccgmccrrr-', 0, 219, 242], [491, 520, 30, '-rcccccccccmccc---------------', 0, 260, 289]],
#[[465, 488, 24, '---------rrrrcccgmccrrr-', 0, 0, 23], [491, 520, 30, '-rcccccccccmccc---------------', 0, 37, 66], [532, 579, 48, '-------------rrrcccccmgccrrrrrrrrrrrrrr---------', 0, 82, 129], [285, 306, 22, '---rrrrrrrrrcccccgcgcr', 0, 134, 155], [313, 352, 40, '-------------rrrrrcccccmcrrrrcmgccmcrrr-', 0, 159, 198], [360, 389, 30, '-------rrrrrrrrcccccmcmccccr--', 0, 204, 233], [409, 427, 19, '-mcmrrrrrrrrrrr----', 0, 236, 254], [433, 459, 27, '------rrrrrrrccgcmccccccrrr', 0, 263, 289]]
]
#mmotifs = [[[940, 940, 1, 'm', 0, 10, 10], [842, 842, 1, 'g', 0, 61, 61], [548, 555, 8, 'rmcccmgr', 0, 83, 90], [297, 305, 9, '---rcmcmc', 0, 130, 138], [342, 346, 5, 'cmmcr', 0, 144, 148], [331, 337, 7, '---rcmc', 0, 155, 161], [375, 384, 10, '---rcmcmcr', 0, 179, 188], [446, 458, 13, 'rcmcmcccgcgcr', 0, 202, 214], [480, 483, 4, 'rcmc', 0, 237, 240], [493, 504, 12, 'rcmgcrrrcgcr', 0, 256, 267], [408, 412, 5, 'rcgcg', 0, 279, 283]]]

# mmotifs = [
# #[[288, 393, 106, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygy-----', 1, 0, 105], [406, 419, 14, 'rrrrgrgrrrrr--', 1, 106, 119], [420, 489, 70, '-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 125, 194], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 208, 244], [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 247, 289]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 93, 202], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 206, 289], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 7, 43],    [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 49, 91]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 94, 203], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 206, 289], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 5, 41],    [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 47, 89]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 95, 204], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 206, 289], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 3, 39],    [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 49, 91]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 96, 205], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 206, 289], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 4, 40],    [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 45, 87]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 49, 158], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 162, 245], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 249, 285], [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 3, 45]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 3, 112],  [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 116, 199], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 203, 239], [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 243, 285]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 91, 200], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 203, 286], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 50, 86],   [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 5, 47]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 91, 200], [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 201, 284], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 1, 37],    [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 43, 85]],
# [[288, 397, 110, '-oyyyyyoroooccgcgc-------------------------oocccmcorrocmmccmc--------------------------oocccmcmccgygyorr------', 1, 3, 112],  [406, 489, 84, 'ocygmymyyyyycc-------------------------occgcmgccccgo--------------------occcmgccc---', 1, 116, 199], [491, 527, 37, '--ccmgcccccgcccocor-----------roocor-', 1, 249, 285], [537, 579, 43, '------yoccccccccmgcc-----------------------', 1, 203, 245]],
# ]

use_predef_start = False

best_seq = "MSPSLFRRLMKNGQKILVVGLPGWQGYTGRQLLEQGHPVRMFNTRSDPERMKKWLERLNVTDIIMHSAMPFGYSKEEAAEALVEAAQELKNRTLIFVMVKTGPGGHHDFFHLNQAEKAAERMVEAIREGKQTGVVHLMGHVNIGHLDTGKWPWPWQPDPEDPQEYIERSIQHYAKVAKKVMKAAKKVDVDYYVFFYVDTWSGPTIADDPRQRKRFFRSLAEDIEKAGPEHGVIWGIEGGHMLDHLEQMLDSPEMLEQHDVGIMYGAPYFPGYLPDKEESARLIKKLVKKL"
#best_seq = "MQIKKGKEVVILSRIIMYFGPSPEQFAEQVEKALEALEKLPEKYIIMVLLKGFGGGGHDTFHYKQIKEEQMERAIEAAQALKKQGLDKTVMVGIHVNASEPSPDYQQDPEMKKRWNWGDKPDKEEMREYHRRFMRRAMRALQKRGFKTMVHIYSDVWGNEQLPGQSEEERQEMRKEFKQTGIYVHTEWNMMEGADHTFVHWYWDTSYASEADAKRVIIIGNSSHMERLLKNGKVLLAPFGWQGFALENAKRVLYLP"

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
sequence_restraint_letters = "gml"
structure_restraint_letters = "rocygm"#"rgmcl"
structure_restraint_mask_values = { "r" : 1, "g" : 4, "m" : 10 , "c" : 3, "l" : 2, "o" : 3, "y" : 8}#{ "r" : 1, "g" : 6, "m" : 11 , "c" : 5, "l" : 2 }

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

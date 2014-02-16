# A_BLOCK_LEN, C_BLOCK_LEN, P_BLOCK_LEN, K_BLOCK_LEN

import os
import sys

# Whether this is distributed or not
isDistributed = True

# Define the parameter values
allVals = [8, 16, 32, 64, 128]
aVals = allVals
cVals = allVals
kVals = allVals
pVals = allVals

# Execute csub script to run make across parameter grid
print 'Launching the Single Tuners'
totalTuners = len(aVals) + len(cVals) + len(kVals) + len(pVals)
curTuner = 1
for curA in aVals:
    for curC in cVals:
        for curK in kVals:
            for curP in pVals:
                print '\tLaunching Tuner ' + str(curTuner) + ' ' + str(totalTuners)
                baseCommand = 'python tuner-matmul_eval-single.py ' + str(curA) + ' ' + str(curC) + ' ' + str(curK) + ' ' + str(curP) 
                if isDistributed:
                    os.system('csub ' + baseCommand)
                else:
                    os.system(baseCommand)
                curTuner += 1
                sys.exit()
                
print 'Completed'
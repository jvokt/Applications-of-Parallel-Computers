# A_BLOCK_LEN, C_BLOCK_LEN

import os
import sys
import numpy

# Whether this is distributed or not
#isDistributed = True
isDistributed = False

# Define the parameter values
aVals = [192, 240, 256, 304, 336];
cVals = [8, 16, 32, 40, 48, 64];

# Execute csub script to run make across parameter grid
print 'Launching the Single Tuners'
totalTuners = len(allVals)
curTuner = 1
for curA in aVals:
    for curC in cVals:
        print '\tLaunching Tuner ' + str(curTuner) + ' of ' + str(totalTuners)
        baseCommand = 'python tuner-matmul_eval-single.py ' + str(curA) + ' ' + str(curC) 
        if isDistributed:
            os.system('csub ' + baseCommand)
        else:
            os.system(baseCommand)
        curTuner += 1
                
print 'Completed'

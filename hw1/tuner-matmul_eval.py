# A_BLOCK_LEN, C_BLOCK_LEN

import os
import sys
import numpy

# Whether this is distributed or not
#isDistributed = True
isDistributed = False

# Define the parameter values
allVals = range(16,512+1,16)

# Execute csub script to run make across parameter grid
print 'Launching the Single Tuners'
totalTuners = len(allVals)
curTuner = 1
for curVal in allVals:
    curA = curVal
    curC = curVal
    print '\tLaunching Tuner ' + str(curTuner) + ' of ' + str(totalTuners)
    baseCommand = 'python tuner-matmul_eval-single.py ' + str(curA) + ' ' + str(curC) 
    if isDistributed:
        os.system('csub ' + baseCommand)
    else:
        os.system(baseCommand)
    curTuner += 1
                
print 'Completed'

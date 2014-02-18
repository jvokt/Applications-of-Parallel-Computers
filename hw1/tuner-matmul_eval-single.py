# A_BLOCK_LEN, C_BLOCK_LEN, K_BLOCK_LEN, P_BLOCK_LEN

import os
import sys
import time

# Wait a few seconds before beginning to prevent file transfer during grid launch
time.sleep(5)

# Get the parameters passed at the command line
curA = int(sys.argv[1])
curC = int(sys.argv[2])

# Form the output filename and executable name
paramName = str(curA) + '_' + str(curC)
outFile = 'tuner-matmul_eval-single-' + paramName + '.csv'
outProg = 'matmul-mine-' + paramName

# Start the initial file
f = open(outFile, 'w')
f.write('PERF\tA_BLOCK_LEN\tC_BLOCK_LEN\tM\tMFLOPS\n')
f.close()

# Execute the commands to build and profile the program
os.system('make clean')
os.system('make A_BLOCK_LEN=' + str(curA) + ' C_BLOCK_LEN=' + str(curC) + ' ' + outProg)
os.system('./' + outProg + ' | grep "PERF" >> ' + outFile)

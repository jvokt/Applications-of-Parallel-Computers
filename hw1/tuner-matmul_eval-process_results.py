# A_BLOCK_LEN, C_BLOCK_LEN, K_BLOCK_LEN, P_BLOCK_LEN

import os
import sys
import glob

# Final output file 
outFilename = 'tuner-matmul_eval-collection.csv'

# Get all files that need to be collected
incrementalOutFiles = glob.glob('tuner-matmul_eval-single-?_?_?_?.csv')

# Load all of data across incremental files
print 'Loading Data Files'
data = []
curFileNum = 1
for curFilename in incrementalOutFiles:
    print '\tData File ' + str(curFileNum) + ' of ' + len(incrementalOutFiles)
    curFileNum += 1
    # Read the entire file
    curFile = open(curFilename, 'r')
    curFileLines = curFile.readlines()
    curFile.close()
    # Process each line in the file (skip first line)
    for curLine in curFileLines[2:]:
        curLineParts = curLine.split('\t')
        curA = int(curLineParts[1])
        curC = int(curLineParts[2])
        curK = int(curLineParts[3])
        curP = int(curLineParts[4])
        curM = int(curLineParts[5])
        curMflops = float(curLineParts[6])
        data.append((curA, curC, curK, curP, curM, curMflops))
    
# Sort the data by megaflops descending
data.sort(key=lambda lineTuple: lineTuple[4], reverse=True)

# Write out the data into one file sorted
# Start the initial collection file
outFile = open(outFilename, 'w')
outFile.write('PERF\tA_BLOCK_LEN\tC_BLOCK_LEN\tK_BLOCK_LEN\tP_BLOCK_LEN\tM\tMFLOPS\n')
# Write data
for dataLine in data:
    outFile.write('PERF\t' + str(dataLine[0]) + '\t' + str(dataLine[1]) + '\t' + str(dataLine[2]) + '\t' + str(dataLine[3]) + '\t' + str(dataLine[4]) + '\t' + str(dataLine[5]) + '\n')
# Close file
outFile.close()

print 'Created Collected Output File'

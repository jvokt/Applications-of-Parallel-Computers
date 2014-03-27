#!/usr/local/bin/python2.7
# encoding: utf-8
'''
Code to plot runtime as the number of particles are scaled
@author:     Phillip Tischler
@copyright:  2014 Phillip Tischler. All rights reserved.
'''

import sys
import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

if __name__ == "__main__":
    # Setup argument parser
    parser = ArgumentParser(description='Plotter for runtime vs. particle count in log-log.', formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(dest='runtimeFile', help='The file to load with runtime information.', nargs=1)
    parser.add_argument(dest='threadCount', type=int, help='The number of threads to use for a line in the plot', nargs='+')
    
    # Process arguments
    args = parser.parse_args()
    
    # Get the arguments from the parsed ones
    runtimeFile = args.runtimeFile[0]
    threadCounts = args.threadCount
    # Display the arguments
    print 'Arguments Received: '
    print '\t Runtime File: ' + runtimeFile
    print '\tThread Counts: ' + str(threadCounts)

    # Load the contents of the file
    dataList = []
    lineNum = 0
    with open(runtimeFile, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            runtime = float(row[1])
            particles = float(row[3])
            threads = float(row[5])
            # Only keep values which are in the thead counts list
            if threads in threadCounts:
                dataList.append([runtime, particles, threads])
    # Sort the list by particles counts
    dataList.sort(key=lambda row: row[1]);
    
    # Set plot properties
    font = {'family' : 'arial',
            'weight' : 'bold',
            'size'   : 8}
    matplotlib.rc('font', **font)
    
    # Make the plot, first create the figure information
    scalingFig = plt.figure(figsize=(6.5,4.5), dpi=150)
    scalingAx = scalingFig.add_subplot(111)
    # Plot each line for each thread
    for curThreadCount in threadCounts:
        # Filter to only get values for line
        dataFilterList = filter(lambda row: row[2]==curThreadCount, dataList)
        if len(dataFilterList) < 2:
            continue
        # Convert to matrix and get x,y values
        dataMat = np.matrix(dataFilterList)
        xVals = dataMat[:,1]
        yVals = dataMat[:,0]
        # Form the ideal values
        yIdeal = yVals
        for pos in range(1,len(yVals)):
            yIdeal[pos] = yIdeal[0] * (xVals[pos] / xVals[0])
        # Plot actual and ideal
        scalingAx.loglog(xVals, yVals, label = (str(curThreadCount) + ' Threads'))
        scalingAx.loglog(xVals, yIdeal, label = (str(curThreadCount) + ' Threads - Ideal'), linestyle='dashed')
    scalingAx.set_title('Particle Scaling: Log-Log of Runtime vs. Number of Particles')
    scalingAx.set_xlabel('Log of Number of Particles')
    scalingAx.set_ylabel('Log of Runtime (sec.)')
    scalingAx.legend(loc='upper left')
    plt.savefig('runtime-plot_scaling.png', dpi=150)
    
        
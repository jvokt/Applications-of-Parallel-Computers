#!/usr/local/bin/python2.7
# encoding: utf-8
'''
Code to plot runtime as the number of threads are scaled
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
    parser.add_argument(dest='particleCount', type=int, help='The number of particles to use for a line in the plot', nargs='+')
    
    # Process arguments
    args = parser.parse_args()
    
    # Get the arguments from the parsed ones
    runtimeFile = args.runtimeFile[0]
    particleCounts = args.particleCount
    # Display the arguments
    print 'Arguments Received: '
    print '\t   Runtime File: ' + runtimeFile
    print '\tParticle Counts: ' + str(particleCounts)

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
            if particles in particleCounts:
                dataList.append([runtime, particles, threads])
    # Sort the list by thread counts
    dataList.sort(key=lambda row: row[2]);
    
    # Set plot properties
    font = {'family' : 'arial',
            'weight' : 'bold',
            'size'   : 8}
    matplotlib.rc('font', **font)
    
    # Make the plot of relative scaling, first create the figure information
    relScalingFig = plt.figure(figsize=(6.5,4.5), dpi=150)
    relScalingAx = relScalingFig.add_subplot(111)
    # Plot each line for each thread
    for curParticleCount in particleCounts:
        # Filter to only get values for line
        dataFilterList = filter(lambda row: row[1]==curParticleCount, dataList)
        if len(dataFilterList) < 2:
            continue
        # Convert to matrix and get x,y values
        dataMat = np.matrix(dataFilterList)
        xVals = dataMat[:,2]
        yVals = dataMat[0,0] / dataMat[:,0] 
        # Plot relative scaling
        relScalingAx.plot(xVals, yVals, label = (str(curParticleCount) + ' Particles'))
    relScalingAx.set_title('Thread Scaling: Plot of Relative Runtime Speedup vs. Number of Threads')
    relScalingAx.set_xlabel('Number of Threads')
    relScalingAx.set_ylabel('Runtime Speedup (Relative to 1 Thread)')
    relScalingAx.legend(loc='upper left')
    plt.savefig('runtime-plot_speedup.png', dpi=150)
    
        
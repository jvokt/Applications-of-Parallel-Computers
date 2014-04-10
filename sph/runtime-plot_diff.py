#!/usr/local/bin/python2.7
# encoding: utf-8
'''
Code to plot relative speeds to highlight overhead costs
@author:     Phillip Tischler
@copyright:  2014 Phillip Tischler. All rights reserved.
'''

import sys
import os
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Set the params to plot
    serial = (317.74, "No OMP");
    parallelNoLocks = (316.45, "OMP w/o Locks")
    parallel = (319.816, "OMP w/ Locks")
    naive = (2302.64, "Naive (Base)")
    
    # Group the values
    overheadData = [serial, parallelNoLocks, parallel]
    comparisonData = [naive, parallel]

    # Set plot properties
    font = {'family' : 'arial',
            'weight' : 'bold',
            'size'   : 8}
    matplotlib.rc('font', **font)
    
    # Make overhead plot
    overheadFig = plt.figure(figsize=(6.5,4.5), dpi=150)
    overhead = overheadFig.add_subplot(111)
    count = 0
    labels = []
    for (curRuntime, curLabel) in overheadData:
        overhead.bar(count, curRuntime, label=curLabel, align="center")
        labels.append(curLabel)
        count += 1
    overhead.set_xticks(range(len(overheadData)))
    overhead.set_xticklabels(labels)
    overhead.set_title('Runtimes Showing Overhead')
    overhead.set_xlabel('Configuration')
    overhead.set_ylabel('Runtime (sec.)')
    plt.savefig('runtime-plot_overhead.png', dpi=150)
    
    # Make comparison plot
    comparisonFig = plt.figure(figsize=(6.5,4.5), dpi=150)
    comparison = comparisonFig.add_subplot(111)
    count = 0
    labels = []
    for (curRuntime, curLabel) in comparisonData:
        comparison.bar(count, curRuntime, label=curLabel, align="center")
        labels.append(curLabel)
        count += 1
    comparison.set_xticks(range(len(comparisonData)))
    comparison.set_xticklabels(labels)
    comparison.set_title('Runtimes Showing Comparison of Naive & Implementation (1 Thread)')
    comparison.set_xlabel('Configuration')
    comparison.set_ylabel('Runtime (sec.)')
    plt.savefig('runtime-plot_comparison.png', dpi=150)
    
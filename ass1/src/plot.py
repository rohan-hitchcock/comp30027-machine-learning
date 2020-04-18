import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


def numeric_histogram(data, path=None, nbins=20):

    x_min = data.min()
    x_max = data.max()
    
    bucket_size = (x_max - x_min) / nbins

    #convert a data value to the its bucket index
    to_bucket_index = lambda z : math.floor((z - x_min) / bucket_size)

    #convert a bucket index to the center of that bucket
    bucket_index_to_center = lambda i : i * bucket_size + bucket_size / 2 + x_min

    #x_ticks are the center of each bucket
    x_ticks = np.array([bucket_index_to_center(i) for i in range(nbins)]) 


    buckets = np.zeros(nbins)

    for x in data.dropna():
        if x == x_max: 
            buckets[-1] += 1
        else:
            buckets[to_bucket_index(x)] += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x_ticks, buckets, bucket_size, color='cornflowerblue')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close(fig)


OUTDIR = "/home/rohan/Desktop/"

"""
DATASET = "../datasets/wdbc.data"
COLS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] 
NUM_COLS = 32
"""
"""
DATASET = "../datasets/wine.data"
COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
NUM_COLS = 14
"""
"""
DATASET = "../datasets/adult.data"
COLS = [0, 2, 10, 11, 12]
NUM_COLS = 15
"""
DATASET = "../datasets/bank.data"
COLS = [0, 5, 9, 10, 11, 12]
NUM_COLS = 15

df = pd.read_csv(DATASET, names=range(NUM_COLS), na_values=['?'])

os.chdir(OUTDIR)

for c in COLS:
    outfile = str(c) + ".svg"
    numeric_histogram(df[c], path=outfile)

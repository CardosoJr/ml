import pandas as pd 
import numpy as np
from scipy.stats import norm

def ProportionSampleSize(p, N = None, sample_err = 0.02, ci = 0.975, inifinite_population = False):
    zalpha = norm.ppf(ci)
    sample_size = 0
    if inifinite_population:
        sample_size = zalpha ** 2 * p * (1 - p) / (sample_err ** 2)
    else: 
        sample_size = zalpha ** 2 * p * (1 - p) * N / ((sample_err ** 2) * p * (1 - p) + (N - 1) * sample_err ** 2) 

    return sample_size

def MeanSampleSize(pop_sigma,  N = None, sample_err = 0.02, ci = 0.975, inifinite_population = False):
    zalpha = norm.ppf(ci)

    if inifinite_population:
        return (pop_sigma * zalpha) ** 2 / (sample_err ** 2)
    else: 
        return (N * (pop_sigma ** 2) * (zalpha ** 2)) / ((N - 1) * (sample_err ** 2) + (pop_sigma ** 2) * (zalpha ** 2))
        


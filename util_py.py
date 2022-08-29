# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 22:18:52 2022

@author: Mani
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 22:12:39 2022

@author: Mani
"""
import numpy as np
# Module of common utility functions
def gausswin(N,alpha):
    L = N-1
    n = np.arange(0,N)-L/2
    w = np.exp(-(1/2)*(alpha*n/(L/2))**2)
    return w

def get_gausswin(sigma,bin_width):
    # sigma = bin_width * N / (2*alpha)
    # Look at help of gausswin to see how the above was derived

    nStd = 6
    N = np.round(nStd*sigma/bin_width)
    alpha = bin_width*N*1/(2*sigma)
    gw = gausswin(N,alpha);
    gw = gw/np.sum(gw)
    return gw

def std_robust(x):
    # Robust standard deviation
    s = np.median(np.abs((x-np.median(x)))/0.6745)
    return s

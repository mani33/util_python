# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:22:56 2024
Quick helpers. For example - close all open plot windows. These functions are
meant to be called from the iPython console.
@author: maniv
"""
import matplotlib.pyplot as plt

def closeall():
    plt.close('all')

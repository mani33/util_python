# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:38:51 2023

@author: maniv
"""

import numpy as np
from copy import deepcopy
#r = np.random.random((1,10))
def test_my_idea(r):
    c = check_mess(r)
    print("c value",c)
    print("r value:",r)
def check_mess(r):
    v = np.random.rand();
    rv = r
    for i in range(rv.shape[1]):
        rv[0,i]=v
    return rv

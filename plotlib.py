# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:25:16 2024
Module for helpers related to matplotlib plotting
@author: Mani Subramaniyan
"""
# All units are in points (1/72 of an inch)
class Params():
    def __init__(self,**kwargs):        
        self.grid_size = [11] # [nRow nCol]
        self.axis_w = 4*72 # width of a single axes
        self.axis_h = 3*72 # height of a single axes
        self.tick_length = 5
        self.tick_pad = 5
        self.axes_fontsize = 9
        self.xlabel_pad = 5
        self.ylabel_pad = 5
        self.title_fontsize = 9
        self.suptitle_fontsize = 12
        self.title_pad = 5
        self.inter_axes_w = 30
        self.inter_axes_h = 30
        self.sup_title_space = 30
        self.sup_xylabel_fontsize = 30
        self.sup_ylabel_right_space = 30
        self.title_row_ind = []
        self.ylabel_col_ind = []
        self.xlabel_row_ind = []
        self.sup_title = False
        self.sup_xlabel = False
        self.sup_ylabel = False
        self.sup_ylabel_right = False
         
def get_figsize(params):
    # Get figure size in inches
    p = params
    axes_w = (p.axes_fontsize * len(p.ylabel_col_ind)) + p.ylabel_pad + \
                p.axes_fontsize + p.tick_pad + p.tick_length + p.axis_w
              
    
    
p = Params()
get_figsize(p)
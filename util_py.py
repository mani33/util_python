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
#%% Module of common utility functions
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

def find_repeats(x,b,splice_gap=0):
    """    
    find_repeats(x, b, splice_gap)
    Find the starting and ending indices of subarrays where a given value (b)
    is repeated in the vector x. Set b = NaN if repeats of NaN's is desired.
    
    Inputs:
        x - numpy vector (1d or 2d)
        b - scalar
        splice_gap - scalar
    Outputs:
        start_ind - 1d numpy array of int; starting index position of repeats
        end_ind   - 1d numpy array of int; ending index position of repeats
    
    Note that single occurrence of b with non-b values before and after will 
    also be included in finding. See Example1 below. Such occurrences will 
    have start_ind=end_ind, allowing for their easy post-hoc elimination 
    if desired.
    
    Optionally, two adjacent subarrays will be spliced if the gap is 
    <= splice_gap (integer value). Note the gap is the difference between the 
    indices of the end of one segment and the beginning of the next segment; 
    therefore, the number of elements in between the apposing edges will be 
    splice_gap-1.
    
    Example 0: find_repeats([NaN 0 0 NaN NaN NaN 0 1 0 0], NaN) will return 
    start_ind = [0,3,7] and end_ind = [0,5,7] as indices of repetitions of NaN.
    
    Example 1: find_repeats([1 0 0 1 1 1 0 1 0 0], 1) will return 
    start_ind = [0,3,7] and end_ind = [0,5,7] as indices of repetitions of 1.
    
    Example 2: find_repeats([11,52,30,4,4,4,5,8,9], 4) - will return 
    start_ind = 3 and end_ind = 5 as indices of repetitions of the value 4.
    
    Example 3: find_repeats([11,52,30,4,4,5,4,4,4,17,29], 4, 2) - will return 
    start_ind = 3 and end_ind = 8 as indices of repetitions of the value 4
    because the gap of 2 between the two subarrays of 4 will now be ignored.
    
    Example 4: find_repeats([11,52,30,4,4,5,9,4,4,4,17,29], 4, 3) will return 
    start_ind = 3 and end_ind = 9 as indices of repetitions of the value 4
    because the gap of 3 between the two subarrays of 4 will now be ignored.
    
    Mani Subramaniyan 2022-12-04
    """
    # Enforce into a 1d array
    x = np.ravel(x)
    xb_ind = np.ravel(np.argwhere(x==b)) # 1d array to make life easier
    if not (xb_ind.size==0):
        di_spliced = np.diff(xb_ind)
        # Replacing "small gaps" with 1 essentially joins the adjacent segments
        di_spliced[di_spliced <= splice_gap] = 1
        # locations where break of continuity occurs
        df_ind = np.ravel(np.argwhere(di_spliced > 1))
        # for the last subarray, need to manually add end location
        # Convert to integers for indexing purposes later on
        end_ind = np.append(xb_ind[df_ind], xb_ind[-1]).astype(int)
        # for the first subarray, need to manually add beginning location
        start_ind = np.append(xb_ind[0],xb_ind[df_ind+1]).astype(int)
    else:
        start_ind = np.array([]).astype(int)
        end_ind = np.array([]).astype(int)
    return start_ind,end_ind

def deviance_logistic(y_true,y_pred):
    """ Compute deviance per sample, for logistic regression
    parameters:
        y_true - 1d numpy array of 1's and 0's, the true class labels
        y_pred - 1d numpy array of predicted class probabilities ((0,1])
    returns:
        d - deviance value per sample  """
    assert y_true.size==y_pred.size, 'y and yhat must be the same length'
    loss_a = np.dot(y_true,np.log(y_pred))
    loss_b = np.dot((1-y_true),np.log(1-y_pred))
    d = -2*(np.sum(loss_a + loss_b))/y_true.size
    return d

def mprint(*args,verbose=True):
    if verbose:
        for v in args:
            print(v,end=" ",flush=True)
def intersect(A,B):
    """ Return intersection of A and B and indices of A and B matching the common
    elements """
    As = set(A)
    Bs = set(B)
    ABi = list(As.intersection(Bs))
    iA = []
    iB = []
    for intElem in ABi:
        iA.append(np.nonzero([a==intElem for a in A])[0])   
        iB.append(np.nonzero([b==intElem for b in B])[0])      
    return ABi,np.hstack(iA),np.hstack(iB)
def false_positive_count(y_true,y_pred,positive_class=1):
    pc = positive_class
    fp = np.nonzero([(yt!=pc)and(yp==pc) for yt,yp in zip(y_true,y_pred)])[0].size
    return fp
def false_negative_count(y_true,y_pred,positive_class=1):
    pc = positive_class
    fn = np.nonzero([(yt==pc)and(yp!=pc) for yt,yp in zip(y_true,y_pred)])[0].size
    return fn
def true_positive_count(y_true,y_pred,positive_class=1):
    pc = positive_class
    tp = np.nonzero([(yt==pc)and(yp==pc) for yt,yp in zip(y_true,y_pred)])[0].size
    return tp
def true_negative_count(y_true,y_pred,positive_class=1):
    pc = positive_class
    tn = np.nonzero([(yt!=pc)and(yp!=pc) for yt,yp in zip(y_true,y_pred)])[0].size
    return tn
def sensitivity(y_true,y_pred,positive_class=1):
    tp = true_positive_count(y_true, y_pred,positive_class=positive_class)
    fn = false_negative_count(y_true, y_pred,positive_class=positive_class)
    sn = tp/(tp+fn)
    return sn
def specificity(y_true,y_pred,positive_class=1):
    tn = true_negative_count(y_true,y_pred,positive_class=positive_class)
    fp = false_positive_count(y_true, y_pred,positive_class=positive_class)
    sp = tn/(tn+fp)
    return sp
def positive_predictive_value(y_true,y_pred,positive_class=1):
    tp = true_positive_count(y_true, y_pred,positive_class=positive_class)
    fp = false_positive_count(y_true, y_pred,positive_class=positive_class)
    if (tp+fp) == 0:
        print(y_true)
        print(y_pred)
    ppv = tp/(tp+fp)
    return ppv
def adj_positive_predictive_value(y_true,y_pred,prevalence,positive_class=1):
    """ Useful when you want to compute PPV for a hypothetical prevalence rate 
        prevalence - probability of a condition (or disease) in the population
    """    
    pr = prevalence
    sn = sensitivity(y_true, y_pred,positive_class=positive_class)
    sp = specificity(y_true, y_pred,positive_class=positive_class)
    dn = ((sn*pr)+((1-sp)*(1-pr)))
    if dn==0:
        print('_______________')
        print(sn,pr,sp)
        print('_______________')
    appv = (sn*pr)/dn
    return appv
def accuracy(y_true,y_pred,positive_class=1):
    tp = true_positive_count(y_true, y_pred,positive_class=positive_class)
    tn = true_negative_count(y_true, y_pred,positive_class=positive_class)
    acc = (tp+tn)/len(list(y_true))
    return acc

def arange(start,stop,step,n_decimals):
    # n_decimals - number of decimals after the decimal point to which we should
    # round the values to check if they cross the stop value. Actual output
    # array values will NOT be rounded    
    assert start < stop, 'start value must be lower than stop value'
    assert step > 0, 'step value must be positive'
 
    enough = False
    v = [start]
    while not enough:
        v.append(v[-1]+step)
        if np.round(v[-1],n_decimals)>stop:
            enough = True
            v = v[0:-1] 
    return np.array(v)
                
def create_psth_bins(pre,post,bin_width,n_decimals=5):
    """
    Create bin edges for peri-stimulus time histograms (psth). The bin edges will
    be guranteed to have zero as one of the egde points.
    
    Inputs:        
        pre: prestimulus time; a negative number in sec
        post: poststimulus time; a positive number in sec
        bin_width: in sec
        n_decimals - number of decimals after the decimal point to which we should
                    round the values to check if they cross the stop value. Actual output
                    array values will NOT be rounded
    Output:
        bin_edges: 1d-numpy array of bin edges in sec
        bin_cen: 1d-numpy array of bin centers in sec
    """
    
    """ Make sure that 0 is included in the bin edges """
    # Keep 0 at starting point; otherwise np.arange may or may not include 0
    # depending on value of pre.
    assert pre < 0, "pre time must be strictly negative"
    assert post > 0, "post time must be strictly positive"
    bins_pre = -np.flipud(arange(0,-pre,bin_width,n_decimals))
    bins_post = arange(0,post,bin_width,n_decimals)
    bins_post = bins_post[1:None] # To avoid including zero twice later when concatenating
    bin_edges = np.concatenate((bins_pre,bins_post))
    bin_cen = bin_edges[:-1]+bin_width/2
    return bin_edges,bin_cen
        
def make_axes(plt,w,h,dpi=300):
    """ Create a new figure, and make a single subplot with axis size w x h in inches """
    # Add space for x and y ticks and labels
    # Note: axis width and height only includes the box of plotting area, 
    # not ticks, tick labels, axis labels etc
    pad_h_tot = 1 # inches of total padding on left and right side
    pad_w_tot = 1 # inches of total padding on top and bottom
    # Add padding to axis width
    fw = w+pad_w_tot
    fh = h+pad_h_tot
    
    p_left = pad_h_tot*0.65/fw
    p_bottom = pad_w_tot*0.65/fh
    
    figsize = [fw,fh]
    aw = w/fw
    ah = h/fh
    fig = plt.figure(figsize=figsize,dpi=dpi)
    ax_pos = [p_left,p_bottom,aw,ah]
    ax = fig.add_axes(ax_pos)
    return fig,ax
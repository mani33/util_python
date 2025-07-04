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
import sklearn.metrics as skm
from itertools import combinations
import pickle as pkl
import scipy.stats as stat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from contextlib import closing
from zipfile import ZipFile
import os
import pingouin as pg
import scipy.cluster.hierarchy as sch
import pandas as pd
import pyperclip
from screeninfo import get_monitors
import PyPDF2
from reportlab.pdfgen import canvas
from PyPDF4.pdf import PdfFileReader, PdfFileWriter
from reportlab.lib.units import mm
#%% Functions
def zscore(x):
    z = (np.array(x)-np.mean(x))/np.std(x)
    
    return z

def get_cdf(data, zscored=True, cdf_x=None):
    """ Compute cumulative distribution function
        Input:
            data - list or 1d numpy array
            n - number of cdf values to be computed between the min and max
                of the data
        Output:
            cdf - cdf values; 1d numpy array of size n
    """    
    # Sort the data
    data = np.sort(data)
    if zscored:
        data = (data-np.mean(data))/np.std(data)
    if cdf_x is None:
        cdf_x = np.unique(data)
        
    cdf = np.searchsorted(data, cdf_x, side='right')/data.size
    
    return cdf, cdf_x
    

def bin_by_time(x_v, x_t, bin_cen_t):
    """ Bin a given 2d array of time series data x_v (n, x_t.size) using the
    bin centers vector bin_cen_t. x_t is the time vector corresponding to the
    raw unbinned data x_v
    Inputs:
        x_v - 2d numpy array of size (p, nT)
        x_t - 1d numpy array of time (nT, )
        bin_cen_t - 1d numpy array of bin center times (mT,); assumed to be 
                    uniformly spaced
    Outputs:
        b_v - 2d numpy array of binned data (p, mT)
    
    Mani Subramaniyan 2025-03-23    
    """
    # Get bin width from the given bin_cen_t
    bw = np.diff(bin_cen_t[0:2])
    # Get start and end times of bins
    bin_s, bin_e = bin_cen_t - bw/2, bin_cen_t + bw/2
    # Get a boolean array telling us which data elements belong to each bin
    bin_bools = [(x_t >= s) & (x_t < e) for s, e in zip(bin_s, bin_e)]
    # Get number of data points in each bin
    n_per_bin = np.sum(bin_bools[0]) 
    # For quick selection of data points falling in each bin, use matrix multiplication
    bin_selector = np.array(bin_bools, dtype=float).T # (nT, mT)
    bin_sums = np.matmul(x_v, bin_selector) # (p, nT) x (nT, mT) = (p, mT)
    b_v = bin_sums/n_per_bin
    
    return b_v

def shuffle_save_pdf_pages(pdf_path, output_file_name):
    """Reads a PDF file and saves each page as a separate PDF."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        pdf_writer = PyPDF2.PdfWriter()
        np.random.seed(1)
        page_order = np.random.randint(0,num_pages,num_pages)
        # Math
        # breakpoint()
        # ndone = 100 # Math
        ndone = 57 # English
        pages_done = page_order[:ndone]
        page_order = list(set(np.arange(num_pages)) - set(pages_done))
        # Shuffle left over pages
        np.random.shuffle(page_order)
        
        for page_num in page_order:
            pdf_writer.add_page(pdf_reader.pages[int(page_num)])       
        with open(output_file_name, 'wb') as output_file:
            pdf_writer.write(output_file)
            
def create_page_pdf(num, tmp, paper_type='letter'):
    c = canvas.Canvas(tmp)
    for i in range(1, num + 1):
        # c.drawString((210 // 2) * mm, (4) * mm, str(i)) # Bottom
        if paper_type == 'letter':        
            c.drawString(6 * mm, (279-12) * mm, str(i)) # Letter paper size
        elif paper_type == 'legal':
            c.drawString(3 * mm, (356-6) * mm, str(i)) # Legal paper size
        c.showPage()
    c.save()

def add_page_numbers(pdf_path, paper_type='letter'):
    """
    Add page numbers to a pdf, save the result as a new pdf
    @param pdf_path: path to pdf
    """
    tmp = "__tmp.pdf"

    writer = PdfFileWriter()
    with open(pdf_path, "rb") as f:
        reader = PdfFileReader(f, strict=False)
        n = reader.getNumPages()

        # create new PDF with page numbers
        create_page_pdf(n, tmp, paper_type=paper_type)

        with open(tmp, "rb") as ftmp:
            number_pdf = PdfFileReader(ftmp)
            # iterarte pages
            for p in range(n):
                page = reader.getPage(p)
                numberLayer = number_pdf.getPage(p)
                # merge number page with actual page
                page.mergePage(numberLayer)
                writer.addPage(page)

            # write result
            if writer.getNumPages():
                newpath = pdf_path[:-4] + "_numbered.pdf"
                with open(newpath, "wb") as f:
                    writer.write(f)
        os.remove(tmp)
     
def get_binary_comb(n_bits,reverse_strings=True):
    # Get a list of strings of binary combinations
    # Loop through all numbers from 0 to 2^n_bits - 1
    bn = []
    for i in range(1 << n_bits):
        # Convert the current number to a binary string of length n
        binary_str = format(i, '0' + str(n_bits) + 'b')
        if reverse_strings:
            binary_str = binary_str[::-1]
        bn.append(binary_str)        
    return bn

def mkdir(dir_name):
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name,exist_ok=True)
def colvec(one_dim_array):
    # Convert one dimentional numpy array to 2d column vector
    return np.reshape(one_dim_array,(-1,1))
    
def rsquared_lmm(mod):
    # Compute R2 as suggested by Nakagawa & Schielzeth (2012)
    # Input mod is a fitted linear mixed model
    var_resid = mod.scale
    # sum the diagonal elements of the mod.cov_re, which correspond to the
    # variances of the random effects. We ignore covariances (the off-diagonal
    # elements)
    var_re = np.diag(mod.cov_re.values).sum() 
    var_fe = mod.predict().var()
    tot_var = var_resid + var_re + var_fe
    r2m = var_fe/tot_var # marginal
    r2c = (var_fe+var_re)/tot_var # conditional
    r2 = dict(marginal=r2m,conditional=r2c)
    
    return r2

def rmse(y,yh):
    # Root mean squared error between y and yh
    return np.sqrt(np.mean((np.ravel(y)-np.ravel(yh))**2))

def get_figure_position():
    # Returns a tuple of the pixel position (x,y,dx,dy) of the current figure
    x_y_dx_dy = plt.get_current_fig_manager().window.geometry().getRect()
    pyperclip.copy(f'utpy.set_figure_position({x_y_dx_dy})')
    return x_y_dx_dy

def set_figure_position(rect):
    # Set the position of the current figure by the given tuple (rect) of new
    # position. rect is a list or tuple of 4 elements: x,y,dx,dy
    plt.get_current_fig_manager().window.setGeometry(*rect)
    
def move_figure(x,y):
    # Move current figure's top left corner to the given (x,y) position
    rect = get_figure_position()
    set_figure_position([x,y,rect[2],rect[3]])
    
def box_off(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def format_pstr(p):
    if p < 0.001:
        ps = 'p < 0.001'
    else:
        ps = 'p = %0.3f'%p
        
    return ps


def scatter_equal(v1, v2, s=4, c='k', xy_lim=None, title=None, ax=None, facecolors='none'):
    v1,v2 = np.ravel(np.array(v1)),np.ravel(np.array(v2))
    if ax == None:
        ax = plt.gca()    
    ax.scatter(v1, v2, s=s, zorder=1, facecolors=facecolors, edgecolors=c)    
    pv = np.hstack((v1,v2))
    if xy_lim==None:
        m,ma = np.min(pv),np.max(pv)
    else:
        m,ma = xy_lim[0],xy_lim[1]

    plt.axis('image')
    if title != None:
        ax.set_title(title)
    plt.tight_layout()
    ax.set_xlim((m,ma))
    ax.set_ylim((m,ma))
    ax.plot([m,ma],[m,ma],color='r',zorder=2)
    
    return ax

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx],idx

def plot_corr_mat(r,feature_names,show_rval=False,abs_corr=False):
    # Plot correlation matrix as an image with locations named by the given
    # feature list.
    plt.figure()
    vmin,vmax = -1,1
    cmap='coolwarm'
    if abs_corr: 
        r = np.abs(r)
        vmin=0
        cmap = 'inferno'
        
    plt.imshow(r,cmap=cmap,vmin=vmin,vmax=vmax)
    plt.colorbar()
    ticks = np.arange(r.shape[0])
    ax = plt.gca()
    ax.set_yticks(ticks,labels=feature_names)
    ax.set_xticks(ticks,labels=feature_names,rotation=90)                 
    plt.tight_layout()
    return ax
        
def rm_corr_mat(df,features,grouping_var):
    # Compute repeated measures correlation between all pairs of features using
    # Bakdash & Marusich 2017 paper implemented in pingouin package
    features = np.array(features)
    corr = np.ones((features.size,features.size))
    for i,fi in enumerate(features):
        for j,fj in enumerate(features):
            if j > i:
                corr[i,j] = float(pg.rm_corr(data=df,x=fi,y=fj,subject=grouping_var)['r'])
            elif i > j: # copy from symmetric location
                corr[i,j] = corr[j,i]
                
    return corr
                
def robust_p2p(x,lower_q_th=0.01,upper_q_th=0.99):
    # Use percentiles to compute peak to peak height to ignore outliers
    q = np.quantile(x,[lower_q_th,upper_q_th])
    p2p = np.abs(np.diff(q))[0]
    
    return p2p

def gausswin(N,alpha):
    L = N-1
    n = np.arange(0,N)-L/2
    w = np.exp(-(1/2)*(alpha*n/(L/2))**2)
    return w

def get_gausswin(sigma,bin_width):
    # sigma = bin_width * N / (2*alpha)
    # Look at help of gausswin of MATLAB to see how the above was derived
    nStd = 6
    N = np.round(nStd*sigma/bin_width)
    alpha = bin_width*N*1/(2*sigma)
    gw = gausswin(N,alpha);
    gw = gw/np.sum(gw)
    return gw

def get_pickled_data(data_filename):
    with open(data_filename,'rb') as fh:
        mdata = pkl.load(fh)
    return mdata

def pickle_data(mdata,save_filename):
    with open(save_filename,'wb') as fh:
        pkl.dump(mdata,fh)      
    
def std_robust(x):
    # Robust standard deviation
    s = np.median(np.abs((x-np.median(x)))/0.6745)
    return s

def find_contiguous_segs(x,g):
    """Find beginning and ending indices of contiguous segments, which are 
    defined to be any sequence of elements with a gap of g or less.
    Inputs:
        x - 1d list or array
        g - float specifying the maxmimum gap between adjacent elements of x
    Outputs:
        start_ind - 1d numpy numpy array of int; starting (inclusive) index position of segments
        end_ind   - 1d numpy numpy array of int; ending (inclusive) index position of segments
    """
    x = np.array(x)
    dx = np.diff(x)
    bdx = (dx <= g).astype(float)
    s1,s2 = find_repeats(bdx, 1)
    s2 = s2+1
   
    return s1,s2
    
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
        start_ind - 1d numpy array of int; starting index (inclusive) position of repeats
        end_ind   - 1d numpy array of int; ending index (inclusive) position of repeats
    
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

def deviance_logistic(y_true,y_pred,class_weights):
    """ Compute deviance per sample, for logistic regression
    parameters:
        y_true - 1d numpy array of 1's and 0's, the true class labels
        y_pred - 1d numpy array of predicted class probabilities ((0,1])
        class_weights - dictionary of class weights. eg. {0:1.5,1:1} to set 
                        1.5:1 weight ratio for classes 0:1
                                                                  
    returns:
        d - deviance value per sample  """
    assert y_true.size==y_pred.size, 'y and yhat must be the same length'
    assert np.unique(y_true).size==2, 'deviance is not implemented for multiclass'
    loss_1 = np.dot(y_true,np.log(y_pred)*class_weights[1])
    loss_0 = np.dot((1-y_true),np.log(1-y_pred)*class_weights[0])
    d = -2*(np.sum(loss_1 + loss_0))/y_true.size
    return d

def mprint(*args,verbose=True):
    if verbose:
        for v in args:
            print(v,end=" ",flush=True)
def intersect(A,B,logical_indices=False):
    """ Return intersection of A and B and indices of A and B matching the common
    elements. Set logical_indices to True if return logical indices of A and B. """
    As = set(A)
    Bs = set(B)
    ABi = list(As.intersection(Bs))
    iA = []
    iB = []
    for intElem in ABi:
        iA.append(np.nonzero([a==intElem for a in A])[0])   
        iB.append(np.nonzero([b==intElem for b in B])[0])
    iA,iB = np.hstack(iA),np.hstack(iB)
    if logical_indices:
        tiA,tiB = np.zeros_like(A,dtype=bool),np.zeros_like(B,dtype=bool)
        tiA[iA],tiB[iB] = True,True
        iA,iB = tiA,tiB
    return ABi,iA,iB
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

def format_figure(plt,**kwargs):
    params = {}
    params['linewidth'] = 0.5
    params['axes_linewidth'] = 0.5
    params['xmargin'] = 0.01
    params['font_name'] = 'Arial'
    params['fontsize'] = 9
    params['nondata_col'] = [0.15,0.15,0.15]
    params['labelpad'] = 12
    for key,v in kwargs.items():
        if key in params.keys():
            params[key] = v
       
    plt.rcParams['axes.xmargin'] = params['xmargin'] # get rid of extra spacing at the edges of x axis
    plt.rcParams['font.family'] = params['font_name']
    plt.rcParams['font.size'] = params['fontsize']
    plt.rcParams['axes.edgecolor'] = params['nondata_col']
    plt.rcParams['xtick.color'] = params['nondata_col']
    plt.rcParams['xtick.labelcolor'] = params['nondata_col']
    plt.rcParams['ytick.color'] = params['nondata_col']
    plt.rcParams['ytick.labelcolor'] = params['nondata_col']
    plt.rcParams['text.color'] = params['nondata_col']
    plt.rcParams['axes.labelcolor'] = params['nondata_col']
    plt.rcParams['legend.labelcolor'] = params['nondata_col']
    plt.rcParams['legend.fontsize'] = params['fontsize']
    plt.rcParams['lines.linewidth'] = params['linewidth']
    plt.rcParams['axes.linewidth'] = params['axes_linewidth']
    plt.rcParams['xtick.major.width'] = params['axes_linewidth']
    plt.rcParams['xtick.minor.width'] = params['axes_linewidth']
    plt.rcParams['ytick.major.width'] = params['axes_linewidth']
    plt.rcParams['ytick.minor.width'] = params['axes_linewidth']
    plt.rcParams['axes.labelpad'] = params['labelpad']
    
def make_axes(plt, wh, dpi=300):
    """ Create a new figure, and make a single subplot with axis size w x h in inches """
    # Add space for x and y ticks and labels
    # Note: axis width and height only includes the box of plotting area, 
    # not ticks, tick labels, axis labels etc
    pad_h_tot = 1 # inches of total padding on left and right side
    pad_w_tot = 1 # inches of total padding on top and bottom
    # Add padding to axis width
    w = wh[0]
    h = wh[1]
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
def get_all_combinations(items):
    """ Get all possible combinations of the given items 
    Input: items - list of items
    Output: cc - list of all possible combinations of the elements of the items
    example: get_all_combinations(['a','b']) will output
    [['a'],['b'],['c'],['a','b'],['a','c'],['b','c'],['a','b','c']]
    """
    assert type(items)==list,'input "items" must be a list'
    k_values = np.arange(len(items))+1 # k in "n choose k"
    cc = []
    for k in k_values:
        cb = list(combinations(items,k))
        for c in cb:
            cc.append(list(c))
    return cc
        
def remove_list_elements(input_list,items_to_remove):
    itr = np.ravel([items_to_remove]) # To handle single element in items_to_remove
    output_list = [x for x in input_list if x not in set(itr)]
    return output_list
    
def accuracy_multiclass(cms):
    """ Calculate accuracy for multiclass
     Inputs:
         cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                         of dict (TP,TN,FP,FN,support)
    Output:
        accuracy """
        
    # Correctly identified entries are TP of all classes (every class is a "positive"
    # class in multiclass classification)
    sv = []
    acc_each_class_ovr = {} # one-versus-rest accuracy for each class
    nPerClass = [] # number of samples in each class
    for key,cc in cms.items():
        sv.append(cc['TP'])
        nPerClass.append(cc['support'])
        acc_each_class_ovr[key] = (cc['TP']+cc['TN'])/(cc['TP']+cc['TN']+cc['TP']+cc['FP'])
    accuracy = np.sum(sv)/np.sum(nPerClass)
    return accuracy, acc_each_class_ovr
    
def get_confusion_matrix(y_true,y_pred,unique_class_labels):
    """ Confusion matrix - following sklearn/Wikipedia convention, the top label of the 
        matrix is Predicted and left side label is True class. """
    nClass = unique_class_labels.size
    cm = np.zeros((nClass,nClass))
    for iRow,true_label in enumerate(unique_class_labels):
        for jCol,pred_label in enumerate(unique_class_labels):
            bc = sum((y_true==true_label) & (y_pred==pred_label)) # equiv to np.logical_and
            cm[iRow,jCol] = bc
    return cm
   
def get_confusion_matrix_summary(cm,unique_class_labels):
    """ Inputs:
    cm - confusion_matrix - np array of nClassesTrue-by-nClassesPredicted
    Outputs:
        cms: dict of (len=nclasses,key=class_label) of dict (TP,TN,FP,FN,support) """
    nClasses = cm.shape[0]
    cms = { }
    all_class_ind = np.arange(0,nClasses)
    for iClass,uClass in enumerate(unique_class_labels):
        iClassInd = np.array([iClass])
        sd = { }
        # Row or Column indices without (wo) the iClass
        rc_ind_wo_iClass = np.ravel(remove_list_elements(all_class_ind,iClass))       
        # True positives------------------------------------
        sd['TP'] = cm[iClass,iClass] # Diagonal elements
        # True negatives------------------------------------
        tn_elements = cm[np.ix_(rc_ind_wo_iClass,rc_ind_wo_iClass)]
        sd['TN'] = np.sum(tn_elements)
        # False positives------------------------------------
        fp_elements = cm[np.ix_(rc_ind_wo_iClass,iClassInd)]
        sd['FP'] = np.sum(fp_elements)
        # False negatives
        fn_elements = cm[np.ix_(iClassInd,rc_ind_wo_iClass)]
        sd['FN'] = np.sum(fn_elements)
        # Number of samples in each class
        sd['support'] = np.sum(cm[iClass,:])
        cms[uClass] = sd
    return cms  
    
def sensitivity_multiclass(cms,average='macro'):
    """ 
    Inputs:
        cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                        of dict (TP,TN,FP,FN,support)
        average - 'micro' or 'macro'
    Outputs:
        sen - average sensitivity
        sv - dict of length nClasses; sensitivity of individual classes
    
    """
    sen_each_class = {}
    sv = []
    match average:
        case 'macro':
            # Get sensitivity of each class and then average across classes           
            for key,cc in cms.items():
                v = (cc['TP']/(cc['TP']+cc['FN']))
                sv.append(v)
                sen_each_class[key] = v
            sen = np.mean(sv)
        case 'micro':
            # Sum up all TP and FN across classes before computing sensitivity
            TP = []
            FN = []
            for _,cc in cms.items():
                TP.append(cc['TP'])
                FN.append(cc['FN'])
            tp_tot = np.sum(TP)
            fn_tot = np.sum(FN)
            sen = tp_tot/(tp_tot+fn_tot)
        case _:
                raise ValueError('%s is not implemented'%average)

    return sen,sen_each_class


def specificity_multiclass(cms,average='macro'):
    """ 
    Inputs:
        cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                        of dict (TP,TN,FP,FN,support)
        average - 'micro' or 'macro'
    Outputs:
        spe - average specificity
        sv - dict length nClasses; specificity of individual classes
    """
    spe_each_class = {}
    sv = []
    match average:
        case 'macro':
            # Get specificity of each class and then average across classes            
            for key,cc in cms.items():               
                v = cc['TN']/(cc['TN']+cc['FP'])
                sv.append(v)
                spe_each_class[key]=v
            spe = np.nanmean(sv)
        case 'micro':
            # Sum up all TN and FP across classes before computing specificity
            TN = []
            FP = []
            for cc in cms.values():
               TN.append(cc['TN'])
               FP.append(cc['FP'])
            tn_tot = np.sum(TN)
            fp_tot = np.sum(FP)
            spe = tn_tot/(tn_tot+fp_tot)
        case _:
                raise ValueError('%s is not implemented'%average)
    return spe,spe_each_class

def ppv_multiclass(cms,average='macro'):
    """ Positive predictive value also called precision
    Inputs:
        cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                        of dict (TP,TN,FP,FN,support)
        average - 'micro' or 'macro'
    Outputs:
        spe - average ppv
        sv - dict of length nClasses; ppv of individual classes
    """     
    ppv_each_class = {}
    pv = []
    match average:
        case 'macro':
            # Get f1score of each class and then average across classes
            for key,cc in cms.items():
                # Positive predictive value
                v = cc['TP']/(cc['TP']+cc['FP'])
                pv.append(v)
                ppv_each_class[key] = v
            ppv = np.nanmean(pv)
        case 'micro':
            # Sum up all TP and FN across classes before computing sensitivity
            TP = np.sum([x['TP'] for _,x in cms.items()])          
            FP = np.sum([x['FP'] for _,x in cms.items()]) 
            ppv = TP/(TP+FP)
        case _:
                raise ValueError('%s is not implemented'%average)
    return ppv,ppv_each_class

def npv_multiclass(cms,average='macro'):
    """ Negative predictive value
    Inputs:
        cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                        of dict (TP,TN,FP,FN,support)
        average - 'micro' or 'macro'
    Outputs:
        npv - average npv
        sv - list of length nClasses; npv of individual classes
    """
    npv_each_class = {}
    nv = []
    match average:
        case 'macro':
            # Get f1score of each class and then average across classes
            for key,cc in cms.items():
                # Positive predictive value
                v = cc['TN']/(cc['TN']+cc['FN'])
                npv_each_class[key] = v
                nv.append(v)
            npv = np.nanmean(nv)
        case 'micro':
            # Sum up all TP and FN across classes before computing sensitivity
            TN = np.sum([x['TN'] for _,x in cms.items()])          
            FN = np.sum([x['FN'] for _,x in cms.items()]) 
            npv = TN/(TN+FN)
        case _:
                raise ValueError('%s is not implemented'%average)
    return npv,npv_each_class

def f1score_multiclass(cms,average='macro'):
    """ 
    Inputs:
        cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                        of dict (TP,TN,FP,FN,support)
        average - 'micro' or 'macro'
    Outputs:
        f1score - average f1score
        f1_each_class - list of length nClasses; f1score of individual classes    
    
    """  
    f1_each_class = {}
    fv = []
    match average:
        case 'macro':
            # Get f1score of each class and then average across classes            
            for key,cc in cms.items():
                # Sensitivity
                sen = cc['TP']/(cc['TP']+cc['FN'])
                # Precision or positive predictive value
                ppv = cc['TP']/(cc['TP']+cc['FP'])               
                f = 2/((1/sen)+(1/ppv))
                # print('sen,ppv,f value: %0.1f,%0.1f,%0.1f'%(sen,ppv,f))
                # if f==np.nan:
                #     raise ValueError('f1 score was nan')
                f1_each_class[key] = f
                fv.append(f)
            f1score = np.nanmean(fv)
        case 'micro':
            # Sum up all TP and FN across classes before computing sensitivity
            TP = np.sum([x['TP'] for _,x in cms.items()])
            FP = np.sum([x['FP'] for _,x in cms.items()])
            FN = np.sum([x['FN'] for _,x in cms.items()])
            sen = TP/(TP+FN)
            ppv = TP/(TP+FP)
            f1score = 2/((1/sen)+(1/ppv))
        case _:
            raise ValueError('%s is not implemented'%average)
    return f1score,f1_each_class

def auc_multiclass(y_true,yp_pred,unique_class_labels):
    """ Compute ROC-AUC using one-versus-rest strategy and get macro average of
    the AUC of each class
    Inputs:
            y_true - list of true labels of samples
            yp_pred - np array nSamples-by-nClasses of class probabilities
            unique_class_labels - list of class labels with order matching yp_pred columns
       Outputs:
           auc_mean - averag AUC
           auc_each_class - dict; keyed by class label; each class AUC
           fpr_grid - false positive rate coordinates for the average ROC curve
           mean_tpr - averaged true positive rate for the average ROC curve           
           data - dict (lenth=nClasses) of sub-dicts; each sub-dict contains
           'fpr','tpr' and 'th' for one-vs-rest based roc data
        """
        
    # Compute AUC for each class versus rest
    # Set the labels of current class as 1 and the rest as 0
    roc_curve_data = {}
    auc_each_class = {}
    nClasses = len(unique_class_labels)
    for iClass,yp in enumerate(yp_pred.T):
        c = unique_class_labels[iClass]
        roc_curve_data[c] = {}
        yt = np.ones_like(y_true)
        yt[y_true!= c] = 0
        roc_curve_data[c]['fpr'],roc_curve_data[c]['tpr'],roc_curve_data[c]['th'] = skm.roc_curve(yt, yp)
        auc = skm.auc(roc_curve_data[c]['fpr'],roc_curve_data[c]['tpr'])
        auc_each_class[c] = auc
    # Average AUC across the classes
    auc_mean = np.mean([v for v in auc_each_class.values()])
    # Average tp and fp after interpolating onto a common fpr
    fpr_grid = np.linspace(0.0,1.0,100)
    mean_tpr = np.zeros_like(fpr_grid)    
    for c in unique_class_labels:
        xx = roc_curve_data[c]['fpr']
        yy = roc_curve_data[c]['tpr']
        tpri = np.interp(fpr_grid,xx,yy)
        mean_tpr += np.copy(tpri)
        roc_curve_data[c]['tpr_interp'] = np.copy(tpri)
    mean_tpr/=nClasses
    
    return auc_mean, auc_each_class, fpr_grid, mean_tpr, roc_curve_data

def deviance_multiclass(y_true,y_pred_prob):
    """ Deviance per sample"""
    nClass = y_pred_prob.shape[1]
    uClasses = np.unique(y_true)
    pc = 0.0
    dev_each_class = {}
    for iClass in range(nClass):
        uClass = uClasses[iClass]
        sel_class_samples = (y_true==uClass)
        nSamplesInClass = np.count_nonzero(sel_class_samples)
        dc = -2*np.sum(np.log(y_pred_prob[sel_class_samples,iClass]))
        dev_each_class[uClass] = dc/nSamplesInClass
        # For computing all class total deviance:
        pc += dc
    dev = pc/y_true.size
    return dev, dev_each_class
    
def matthews_corrcoef(y_true,y_pred,cms): 
    """
    cms - confusion_matrix_summary; dict of (len=nclasses,key=class_label) 
                                    of dict (TP,TN,FP,FN,support)
    y_true - np array of true labels of samples
    y_pred - np array of predicted labels
    
    """
    uClasses = np.unique(y_true)
    mcc_each_class = {}
    for uClass in uClasses:
        cm = cms[uClass] # User one-versus-rest two-class classification scheme
        top = (cm['TP']*cm['TN'])-(cm['FP']*cm['FN'])
        b1 = cm['TP']+cm['FP']
        b2 = cm['TP']+cm['FN']
        b3 = cm['TN']+cm['FP']
        b4 = cm['TN']+cm['FN']
        b = b1*b2*b3*b4
        bottom = np.sqrt(b)
        mcc_each_class[uClass] = top/bottom
    # For multiclass, just use sklearn
    mcc = skm.matthews_corrcoef(y_true, y_pred)
    return mcc, mcc_each_class
    
def posterior_odds_multiclass(y_true,y_pred,cms):
    """ posterior_odds = (p(D+)/p(D-))*(sen/(1-spe)) where p(D+) is the 
     proportion of positive class in y_true """
    unique_class_labels = np.unique(y_true)
    n_tot = y_true.size
    _,sen_each_class = sensitivity_multiclass(cms)         
    _,spe_each_class = specificity_multiclass(cms)
    post_odds_each_class = {}
    for uClass in unique_class_labels:
        # Prior true class probabilities                
        pd_pos = cms[uClass]['support']/n_tot
        pd_neg = 1.0-pd_pos
        prior_odds = pd_pos/pd_neg
        sen = sen_each_class[uClass]
        spe = spe_each_class[uClass]
        likelihood_ratio = sen/(1-spe)
        post_odds = prior_odds*likelihood_ratio
        post_odds_each_class[uClass] = post_odds
    mean_post_odds = np.nanmean([x for x in post_odds_each_class.values()])
    return mean_post_odds, post_odds_each_class

def post_test_prob(posterior_odds):
    ptp = posterior_odds/(1+posterior_odds)
    return ptp    

def set_common_subplot_params(plt):
    """
    # Set parameters that will be common for all subplots
    Inputs: 
        plt - matplotlib.pyplot object
    Ouputs:
        None
    """
    legend_label_size = 14
    title_size = 14
    x_y_tick_label_size = 14
    tick_len = 3
    line_width= 1
    fontname = 'Arial'
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': [fontname],
        'axes.titlesize': title_size,
        'axes.linewidth': line_width,
        'axes.labelsize': legend_label_size,
        'xtick.labelsize': x_y_tick_label_size,
        'ytick.labelsize': x_y_tick_label_size,
        'xtick.major.size': tick_len,
        'xtick.major.width': line_width,
        'ytick.major.size': tick_len,
        'ytick.major.width': line_width,
        'text.usetex': False,
        'legend.fontsize':legend_label_size
        }
    plt.rcParams.update(params)
                
def roc_auc_ci(auc,n_true_pos,n_true_neg):
    """ Compute 95% Confidence interval for a given value of AUC. Formulas are based
    on Hanley, J.A. and NcNeil, B.J. 1982. 'The Meaning and Use of the Area under
    a Receiver Operating Characteristic(ROC) Curve.' Radiology, Vol 148, 29-36."""
    aucs = auc**2
    q1 = auc/(2-auc)
    q2 = (2*aucs)/(1+auc)
    nu = auc*(1-auc) + (n_true_pos-1)*(q1-aucs) + (n_true_neg-1)*(q2-aucs)
    de = n_true_pos*n_true_neg
    se = np.sqrt(nu/de)
    ciz = stat.norm.interval(confidence=0.95,loc=0,scale=1)
    ci = auc + np.array(ciz)*se
    return ci
    
def find_closest_val_index(x,v):
    # In the given list x, find the location of the value closest to v
    x = np.ravel(np.array(x))
    ind = np.argmin(np.abs(x-v))
    return ind
    
def get_frame(fig):
    """ Create a 2d numpy array of the plots in the given figure
    Inputs:
        fig - matplotlib figure handle or object
    Ouputs:
        im - 3d numpy array of the color image of the given figure
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    im = np.array(canvas.buffer_rgba())
    return im
    
def get_zipped_file_list(zip_file_name):
    # Get list of files inside the zipped file 
    file_list = []
    with closing(ZipFile(zip_file_name)) as arch:
       ar = arch.infolist()      
       file_list = [os.path.basename(x.filename) for x in ar if not x.is_dir()]
    return file_list  

def dos2unix(input_filename,output_filename):
    """ Convert dos linefeeds (crlf) to unix (lf)
    Example usage:
        dos2unix('test.pkl','test.pkl')
    """
    contents = ''
    outsize = 0
    
    with open(input_filename,'rb') as infile:
        contents = infile.read()
    with open(output_filename,'wb') as outfile:
        for line in contents.splitlines():
            outsize+=len(line)+1
            outfile.write(line+str.encode('\n'))
    print('Done. Saved %s bytes.'%(len(contents)-outsize))
            
def polygon_area(x,y):
    """ Compute polygon area based on the Shoelace formula: copied from:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    Inputs:
        x,y - 1d numpy arrays of x and y coordinates; points in x and y must be 
        ordered in the clockwise or counter-clockwise direction in the xy-plane;
        they can't be in random order.
    Output:
        a - area of the given polygon
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    

def make_segments(x, n, p, valid_last_seg=False):
    """
    Segment the given vector into segments with or without overlap.
    
    make_segments(X,N,P) partitions the signal vector X into
    N-element segments overlapping by P elements (set P=0 for contiguous partition).
    The last segment may not have N elements from the segmentation. Set the
    valid_last_seg to true to exclude the last segment that is not full
    length. Otherwise, by default, short last segment will be made to full
    length (N) using NaN's making up for missing data points. Output Y is a
    N-by-Num_Segments 2d numpy array.
    
    Inputs:
        x - 1d numpy array of data to be segmented
        n - int, segment size
        p - int, overlap size
    Output:
        Y - 2d numpy array of shape (n,nSegments)
    """
    assert n > 0, 'N must be > 0'
    assert p >= 0, 'P must be >= 0'
    assert p < n, 'Overlap length(p) must be less than segment length(n)'
    
    n,p = int(n),int(p)
    nx = x.size
    done = False
    Y = []
    i = 0
    while not done:       
        overlap_offset = i * p
        # Idea: For contiguous segmenting, you will use (i-1)*n+1 as the
        # segment starting index. Because of overlap, pull this index to the
        # left by (i-1)*p.
        seg_begin = (i * n) - overlap_offset
        seg_end = seg_begin + n

        # Are we are in the last segment?
        in_last_seg = seg_begin > (nx - n)

        if not in_last_seg:  # Simpler life when NOT in the last segment
            Y.append(x[seg_begin:seg_end])
        else:  # In the last segment!
            done = True
            # Segment end can't go beyond available data
            seg_end = np.min([nx, seg_end])
            seg_data = x[seg_begin:seg_end]            
            if valid_last_seg:
                if seg_data.size == n:  # Take only full length last segments
                    Y.append(seg_data)
            else:  # Fill in with nan if necessary
                n_fill = n - seg_data.size               
                Y.append(np.hstack([seg_data, np.full((n_fill,), np.nan)]))
        i += 1
    return np.array(Y).T
        
def on_click(event):
    # Function to go with make_subplots_selectable(fig)
    if event.inaxes:
        plt.sca(event.inaxes)

def make_subplots_selectable(fig):
    # Set figure so that if you click on a subplot, it will become the current
    # axes, which you can then alter as you wish, such as changing xlim etc
    fig.canvas.mpl_connect('button_press_event', on_click)
    
def jitter(x, noise_std=0.1):
    # Add normally distributed noise to given data
    x = np.array(x)
    x = x + np.random.normal(0,noise_std,size=x.size)
    return x
   
    
def match_xlim_by_xlabel(fig1, fig2):
    """ Match xlimits of subplots with the same xlabel in the given two figure
    numbers.
    Inputs:
       Example: match_xlim_subplots(1,2)
    """
    plt.figure(fig1)
    ax1 = plt.gcf().get_axes()
    xlabels1 = [a.get_xlabel() for a in ax1]
    
    plt.figure(fig2)
    ax2 = plt.gcf().get_axes()
    xlabels2 = [a.get_xlabel() for a in ax2]
    
    _,iA,iB = intersect(xlabels1, xlabels2)
    
    # Find common xlim
    for ia, ib in zip(iA,iB):
        mm = ax1[ia].get_xlim() + ax2[ib].get_xlim()
        com_lim = [np.min(mm), np.max(mm)]
        ax1[ia].set_xlim(com_lim)
        ax2[ib].set_xlim(com_lim)
        
def match_ylim_by_xlabel(fig1, fig2):
    """ Match xlimits of subplots with the same xlabel in the given two figure
    numbers.
    Inputs:
       Example: match_xlim_subplots(1,2)
    """
    plt.figure(fig1)
    ax1 = plt.gcf().get_axes()
    xlabels1 = [a.get_xlabel() for a in ax1]
    
    plt.figure(fig2)
    ax2 = plt.gcf().get_axes()
    xlabels2 = [a.get_xlabel() for a in ax2]
    
    _,iA,iB = intersect(xlabels1, xlabels2)
    
    # Find common xlim
    for ia, ib in zip(iA,iB):
        mm = ax1[ia].get_ylim() + ax2[ib].get_ylim()
        com_lim = [np.min(mm), np.max(mm)]
        ax1[ia].set_ylim(com_lim)
        ax2[ib].set_ylim(com_lim)

class Mfig():
    def __init__(self,axes_wh, row_with_xlabels,
                     col_with_y1_labels, row_with_titles=None, 
                     col_with_y2_labels=None, col_with_twinx=None, 
                     sup_xlabel=False, sup_ylabel=False, sup_title=False,
                     inter_row_gap=0.1, inter_col_gap=0.1,
                     fig_pad_top=0.2, fig_pad_bottom=0.2,
                     fig_pad_left=0.2, fig_pad_right=0.2,
                     xy_label_fontsize=9, xy_label_pad=0.1,                     
                     tick_label_fontsize=9, 
                     tick_len=0.05, tick_label_pad=0.1,
                     title_fontsize=9,
                     title_pad=0.1,
                     axes_linewidth=1, # points
                     sup_xy_label_fontsize=12,
                     sup_title_fontsize=12,
                     ytick_label_max_ndigits = 3,
                     ytick_label_has_decimal_point=True,
                     box_off=False,
                     axis_off=False,
                     dpi=300):
        
        
        self.xy_label_fontsize = xy_label_fontsize
        self.xy_label_pad = xy_label_pad * 72 # convert inches to points
        self.xy_label_fontsize = xy_label_fontsize
        self.tick_label_fontsize = tick_label_fontsize
        self.title_fontsize = title_fontsize
        self.title_pad = title_pad * 72 # convert inches to points
        
        
        # Determine figure width
        n_row = len(list(row_with_xlabels))
        n_col = len(list(col_with_y1_labels))
        pt2in = 1/72 # points to inches conversion factor
        if col_with_y1_labels is None:
            col_with_y1_labels = np.zeros(n_col)
        if col_with_y2_labels is None:
            col_with_y2_labels = np.zeros(n_col)
        if row_with_titles is None:
            row_with_titles = np.zeros(n_row)
        # First calculate width inches
        x_tick_group_len = (tick_label_fontsize * pt2in) + tick_label_pad + tick_len
        ytick_label_width = (ytick_label_max_ndigits + \
                    ytick_label_has_decimal_point) * (0.5 * tick_label_fontsize)
        y_tick_group_len = (ytick_label_width * pt2in) + tick_label_pad + tick_len
        n_ylabels = np.sum(col_with_y1_labels) + np.sum(col_with_y2_labels)
       
        tot_axes_w = n_col * axes_wh[0]
        tot_axes_h = n_row * axes_wh[1]
        n_y2_labels = np.sum(col_with_y2_labels)
        tot_tick_grp_w = (n_col + n_y2_labels) * y_tick_group_len
        tot_tick_grp_h = n_row * x_tick_group_len
        tot_inter_col_gap_w = (n_col-1) * inter_col_gap
        tot_inter_row_gap_h = (n_row-1) * inter_row_gap
        tot_fig_pad_w = fig_pad_left + fig_pad_right
        tot_fig_pad_h = fig_pad_top + fig_pad_bottom
        xy_label_grp_len = ((xy_label_fontsize * pt2in) + xy_label_pad)
        tot_ylab_w = n_ylabels * xy_label_grp_len
        tot_xlab_h = np.sum(row_with_xlabels) * xy_label_grp_len
        one_title_len = (title_fontsize * pt2in) + title_pad
        tot_title_h = np.sum(row_with_titles) * one_title_len
        fig_w = tot_fig_pad_w + tot_ylab_w + tot_tick_grp_w + tot_axes_w + \
                                                        tot_inter_col_gap_w
        fig_h = tot_fig_pad_h + tot_xlab_h + tot_tick_grp_h + tot_axes_h + \
                        tot_inter_row_gap_h + tot_title_h
        # Check monitor resolution
        self.fig = plt.figure(figsize=(fig_w, fig_h),dpi=dpi)
        mon = get_monitors()
        pm = [y for y in mon if y.is_primary][0]
        mw, mh = pm.width, pm.height
        menu_height = self.fig.canvas.toolbar.size().height() # pixels
        non_plottable_pix_h = menu_height * 2 # approximation        
        max_good_dpi_y = np.floor((mh - non_plottable_pix_h)/fig_h)
        max_good_dpi_x = np.floor(mw/fig_w)
        assert dpi <= max_good_dpi_x, f'The given dpi of {dpi} exceeds the '\
                                     f'possible dpi of {max_good_dpi_x}. '\
                                     f'Set dpi <= {max_good_dpi_x} to get '\
                                     'desired axes width'
        assert dpi <= max_good_dpi_y, f'The given dpi of {dpi} exceeds the '\
                                     f'possible dpi of {max_good_dpi_y}. '\
                                     f'Set dpi <= {max_good_dpi_y} to get '\
                                     'desired axes height'        
        offset_b = fig_pad_bottom
        self.axes = []
        for r in range(n_row-1,-1,-1):
            rax = []
            offset_l = fig_pad_left
            for c in range(n_col):
                left = offset_l + (col_with_y1_labels[c] * xy_label_grp_len) \
                                                            + y_tick_group_len
                                                                    
                bottom = x_tick_group_len + (row_with_xlabels[r] * xy_label_grp_len) \
                                                                        + offset_b
                rect = np.array([left, bottom, axes_wh[0], axes_wh[1]])/\
                        np.array([fig_w, fig_h, fig_w, fig_h])
                
                offset_l = left + axes_wh[0] + (col_with_y2_labels[c] * \
                                            (y_tick_group_len + xy_label_grp_len)) + \
                                            inter_col_gap
                ax = self.fig.add_axes(rect)
                ax.tick_params(axis='both', which='major', direction='out', 
                               length=tick_len*72, width=axes_linewidth, color='k', 
                               pad=tick_label_pad*72, 
                               labelsize=tick_label_fontsize, 
                               labelcolor='k')
                ax.spines['top'].set_linewidth(axes_linewidth)
                ax.spines['right'].set_linewidth(axes_linewidth)
                ax.spines['bottom'].set_linewidth(axes_linewidth)
                ax.spines['left'].set_linewidth(axes_linewidth)
                if box_off:
                    box_off(ax)
                if axis_off:
                    ax.axis('off')
                rax.append(ax)
            self.axes.append(rax)
            offset_b = bottom + axes_wh[1] + (row_with_titles[r] * one_title_len) + \
                                        inter_row_gap

        self.axes = self.axes[::-1]
        self.curr_axes = None
    
    def activate_axes(self,r,c):
        self.curr_axes = self.axes[r][c]
        return self.curr_axes
    
    def plot(self, x, y, **kwargs):
        self.curr_axes.plot(x, y, **kwargs)
    def add_patch(self, *kwargs):
        self.curr_axes.add_patch(*kwargs)
    def imshow(self,im, **kwargs):
        self.curr_axes.imshow(im, **kwargs)
    def xlabel(self,label_str, **kwargs):
        self.curr_axes.set_xlabel(label_str, labelpad=self.xy_label_pad,
                                  fontsize=self.xy_label_fontsize, **kwargs)
    def ylabel(self,label_str, **kwargs):
        self.curr_axes.set_ylabel(label_str, labelpad=self.xy_label_pad,
                                  fontsize=self.xy_label_fontsize, **kwargs)
    def axis_on(self):
        self.curr_axes.axis('on')
    def axis_off(self):
        self.curr_axes.axis('off')
    def text(self,x,y,text_str, **kwargs):
        self.curr_axes.text(x, y, text_str, **kwargs)
    def title(self, tit_str, **kwargs):
        self.curr_axes.set_title(tit_str, pad=self.title_pad, 
                                 fontsize=self.title_fontsize, **kwargs)
        

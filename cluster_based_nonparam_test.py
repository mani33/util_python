# -*- coding: utf-8 -*-
"""
Cluster-based nonparametric test for identifying time segments showing 
an effect anywhere a time series. 

This is an implementation of the algorithm of the following reference:
Maris E, Oostenveld R. Nonparametric statistical testing of EEG- and MEG-data. 
J Neurosci Methods. 2007 Aug 15;164(1):177-90. 
doi: 10.1016/j.jneumeth.2007.03.024. Epub 2007 Apr 10. PMID: 17517438.

Mani Subramaniyan, 2022-12-03

"""
import numpy as np
import scipy.stats as sps
import util_py as upy
from copy import deepcopy
import time

def cluster_mass_test(bin_cen_t,r,stat_test,nBoot=2000,**kwargs):
    """ 
    Perform cluster mass test to find contiguous time points that are
    significantly modulated by stimulus.
    Inputs:
      bin_cen_t - 1d numpy array of floats, of length m; bin center of 
                  times (sec) relative to stimulus onset. Must have negative
                  times indicating pre-stimulus time and positive times
                  indicating post-stimulus times. The number of positive and
                  negative time bins must be equal to allow for random
                  partitions during the bootstrapping.
              r - n-by-m numpy array; n is number of trials or subjects; m is number
                  of samples
      stat_test - string; should be one of 't'(t-test) or 'ranksum'
          nBoot - number of random partitionings
      * *kwargs - optional key word-argument pairs. Acceptable key words are:
                  nBoot (nBoot=10000, default)
    Outputs:
      sig_time_ind - 1d numpy array of bin_cen_t-indices of clusters that are 
      significantly modulated
    """                  
    # Logic of the test:
    # Each row in r is a single trial time series or a time series that was
    # averaged across trials for a single subject. Each column of r is a 
    # response value at a given time bin. The time vector is bin_cen_t.
    
    # We will do a statistical test on each time bin. That is, including the
    # baseline time bins. This is possible because, for each time series, we
    # will have the mean of the baseline against which every element of the
    # time series will be compared. So, if you have n time series, then you
    # will have n mean baseline values and n-by-m responses, giving you m
    # number of statistical tests, each corresponding to a time point.
    
    # Once you have the list of t-statistic or other statistic, you select
    # the ones above a threshold that is equivalent to an alpha of 0.05 for a
    # two-sided test (default). Then, cluster the selected bins by their
    # adjacency. That is, the bins should be continguous within a cluster.
    # Then, find the cluster mass, that is, sum the test statistic within 
    # each cluster. For getting a bootstrap null distribution, we will pick 
    # the largest sum during every iteration of the bootstrapping.
    
    # The procedure above will give us a list of clusters that had statistical
    # significance but they were not corrected for multiple comparison. So, in
    # the next step, we will create a null distribution and find which of those
    # clusters are significant with the family-wise error rate taken care of.
    
    # To create a null distribution, as suggested by Maris & Oostenveld 2007,
    # section 5, we will go to each trial or trial-averaged subject data vector
    # and split it into baseline[1-by-(m/2] and post-stimulus onset[1-by-(m/2]
    # segments. Then, independently for each trial, we randomly permute (swap)
    # the baseline and post-stim segments to obtain the shuffled n-by-m time 
    # series. We will then find the cluster mass of the largest cluster in 
    # this dataset. We will repeat this say 10,000 times and get a null 
    # distribution for the cluster mass to find the p-value of the cluster
    # masses of the original dataset.
    
    # Phew!        
 
    # Check inputs
    assert bin_cen_t.ndim==1,'bin_cen_t must be 1d numpy array'
    assert (r.ndim==2) & (np.remainder(r.shape[1],2)==0),'r must be a 2d numpy array and should have even number of columns'
    
    # Step-1: Get cluster-mass stat for the actual data
    _,abs_clus_stat_sums,clus_start_ind,clus_end_ind = \
        get_cluster_mass_stats(bin_cen_t,r,stat_test)
    # Generate null distribution by random partitioning    
    null_dist = []
    for iBoot in range(nBoot):
       sr = shuffle_time_series(r)
       abs_largest_clus_mass,_,_,_ = get_cluster_mass_stats(bin_cen_t,r,stat_test)
       null_dist.append(abs_largest_clus_mass)
       print(iBoot)
    # Find Monte-Carlo p-value of original data clusters
    null_dist = np.array(null_dist)
    nClus = len(abs_clus_stat_sums)
    clus_p_val = []    
    for clus_sum in abs_clus_stat_sums:
        p = (np.argwhere(null_dist > clus_sum).shape[0])/nBoot
        clus_p_val.append(p)
    clus_p_val = np.array(clus_p_val)
    # Get the time indices of significant clusters
    alpha = 0.05/2 # two-sided test
    sig_clus_ind = np.ravel(np.argwhere(clus_p_val < alpha))
    sig_time_ind = []
    if not sig_clus_ind.size == 0:
        for iClus in sig_clus_ind:
            # print(clus_start_ind[iClus])
            time_ind = list(range(clus_start_ind[iClus],clus_end_ind[iClus]+1))
            sig_time_ind.append(time_ind)
    return np.ravel(np.array(sig_time_ind).astype(int))

def shuffle_time_series(r):
    # We will swap the baseline and post-stim segments of each trial
    n, m = r.shape
    mid = int(m/2)    
    # Go to each row, if swap is needed, split it in two and swap
    shuffled_r = deepcopy(r)# This step is a must
    for i in range(n):
        if np.random.binomial(1, 0.5)==1:
            rt = deepcopy(shuffled_r[i,:]);
            shuffled_r[i,0:mid] = rt[mid:None]
            shuffled_r[i,mid:None] = rt[0:mid]
    return shuffled_r

def get_stats_for_timeseries(bin_cen_t,r,stat_test):
    """
    Perform statistical test on each time point    
    Inputs:
      bin_cen_t - 1d numpy array of floats, of length m; bin center of 
                  times (sec) relative to stimulus onset. Must have negative
                  times indicating pre-stimulus time and positive times
                  indicating post-stimulus times. The number of positive and
                  negative time bins must be equal to allow for random
                  partitions during the bootstrapping.
              r - n-by-m numpy array; n is number of trials or subjects; m is number
                  of samples
      stat_test - string; should be one of 't'(t-test) or 'signed_rank'
    Ouputs:
        statistic - 1d (len=n) numpy array of floats; If 't' test, t-statistic 
                    values; if 'signed_rank' test, sum of positive or negative
                    ranks depending on positive or negative significant modulation
          pos_ind - 1d (len=n) numpy array of booleans; true if positively modulated
          neg_ind - 1d (len=n) numpy array of floats; true if negatively modulated
    """            
    n, m = r.shape
    alpha = 0.05 # alpha - Type-1 error threshold  
    # Step 1: Find mean of baseline for each time series
    baseline_means = np.ravel(np.zeros((1,n),dtype=float))

    for i in range(n):
        baseline_means[i] = np.mean(r[i,bin_cen_t<0])
    # Step 2: Perform statistical test at each time point. If significant, find
    # out if positively or negatively modulated
    statistic = np.ravel(np.zeros((1,m),dtype=float))# 1d numpy array of floats
    pos_ind = np.ravel(np.zeros((1,m),dtype=bool)) # 1d numpy array of booleans
    neg_ind = np.ravel(np.zeros((1,m),dtype=bool)) # 1d numpy array of booleans
    match stat_test:
        case 't': # t-test
            for iTime in range(m):               
                tval,pvalue = sps.ttest_rel(r[:,iTime],baseline_means,
                                            alternative='two-sided')
                statistic[iTime] = tval
                if pvalue < alpha:
                    # Because t-test is based on means, we could compare means
                    # to decide on direction of modulation
                    if (np.mean(r[:,iTime])-np.mean(baseline_means))>0:
                        pos_ind[iTime] = True
                    else:
                        neg_ind[iTime] = True
                    # The case of equal means should not arise as we are dealing
                    # only with significantly modulated case
        case 'signed_rank': # Wilcoxon signed-rank test           
            for iTime in range(m):               
                _,pvalue = sps.wilcoxon(r[:,iTime]-baseline_means,
                                        alternative='two-sided')
                print(iTime)
                if pvalue < alpha:
                    
                   # signed rank values will be different depending on which
                   # side of the distribution is tested (greater or less)
                    rk,pvalue = sps.wilcoxon(r[:,iTime]-baseline_means,
                                            alternative='greater')
                       # alpha of 0.05 was used for two sided test, which means that
                       # for each side 0.05/2 was required to pass the test. So when
                       # confirming the direction of the effect, we use alpha/2.
                    if pvalue < (alpha/2):
                       statistic[iTime] = rk
                       pos_ind[iTime] = True
                    else:
                       rk,pvalue = sps.wilcoxon(r[:,iTime]-baseline_means,
                                                alternative='less')
                       assert pvalue < (alpha/2),'error in pvalue calculations'
                       neg_ind[iTime] = True
                       statistic[iTime] = rk
        
        case _:
            raise ValueError("stat_test must be 't' or 'signed_rank'")
    return np.array(statistic),pos_ind,neg_ind

def get_cluster_mass_stats(bin_cen_t,r,stat_test,**kwargs):
    """
    Find clusters of contiguous time bins that have values significantly
    different from the mean of baseline (mean of time bins with 
    center times < 0). Because mean of baseline (a scalar) is used for 
    comparisons, even the baseline period can have clusters that have values 
    significantly different from the mean of baseline.
    
    Inputs:
      bin_cen_t - 1d numpy array of floats, of length m; bin center of 
                  times (sec) relative to stimulus onset. Must have negative
                  times indicating pre-stimulus time and positive times
                  indicating post-stimulus times. The number of positive and
                  negative time bins must be equal to allow for random
                  partitions during the bootstrapping.
      r - n-by-m numpy array; n is number of trials or subjects; m is number
          of samples
      stat_test - string; should be one of 't'(t-test) or 'ranksum'
      **kwargs - optional key word-argument pairs. Acceptable key words are:
                  nBoot (nBoot=10000, default)
    Outputs:
      abs_stat_val_largest_clus - float; mass(sum of absolute value of 
                                    statistic of all members of the cluster) 
                                    of the largest cluster
      abs_clus_stat_sums - 1d numpy array of floats; absolute value of sum of 
                                    stat values of each cluster
      clus_start_ind - 1d numpy int array of bin_cen_t- start indices of clusters
      clus_end_ind - 1d numpy int array of bin_cen_t- end indices of clusters    
    """
    
    # Input check: positive and negative time bins must be equal
    n_neg = np.argwhere(bin_cen_t<0).size
    n_pos = np.argwhere(bin_cen_t>0).size
    assert n_pos==n_neg,\
    'Number of positive time bins (%u) must be equal number of negative time bins (%u)'\
        % (n_pos,n_neg)
    
    # Step-1: Get the test statistic value for each time point
    
    stats,pos_ind,neg_ind = get_stats_for_timeseries(bin_cen_t,r,stat_test) # output: 1d np array
    
    pos_starts,pos_ends = upy.find_repeats(pos_ind.astype(float), 1)
    neg_starts,neg_ends = upy.find_repeats(neg_ind.astype(float), 1)
    # Because positive and negative clusters will not have any overlaps, we can
    # simply pool all the starts and sort them and repeat this for the ends. In
    # the resulting arrays, the starts and ends will be matched automatically.
    clus_start_ind = np.sort(np.concatenate((pos_starts,neg_starts))).astype(int)
    clus_end_ind = np.sort(np.concatenate((pos_ends,neg_ends))).astype(int)
    # Find the cluster with the largest sum of test-statistic. This cluster
    # may or may not be the cluster with largest number of members
    # Sum test-statistics within each cluster
    clus_stat_sums = []
    stat_val_largest_clus = np.array([])
    abs_clus_stat_sums = np.array([])
    if not clus_start_ind.size==0:
        for s,e in zip(clus_start_ind,clus_end_ind):           
            clus_stat_sums.append(np.sum(stats[s:(e+1)]))
        # Find cluster with largest absolute value of stat sum
        abs_clus_stat_sums = np.abs(clus_stat_sums)
        max_ind = np.argmax(abs_clus_stat_sums)        
        stat_val_largest_clus = np.max(abs_clus_stat_sums)
    return stat_val_largest_clus,abs_clus_stat_sums,clus_start_ind,clus_end_ind
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

def cluster_mass_test(bin_cen_t,r,stat_test,nBoot=1000,**kwargs):
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
    # response value at a given time bin. The time vector (bin_cen_t) will 
    
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
    
    # To create a null distribution, we will take each 1-by-m time series of
    # the n-by-m response array, split them into baseline and post-stimulus 
    # segment (1-by-[m/2]). Then, from the 2*n-by-(m/2) time series segments,
    # we will pick two 1-by-[m/2] segments at a time and join them back into
    # 1-by-m segments to obtain the shuffled n-by-m time series. We will then
    # find the cluster mass of the largest cluster in this dataset. We will
    # repeat this say 10,000 times and get a null distribution for the cluster
    # mass to find the p-value of the cluster masses of the original dataset.
    
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
    return np.array(sig_time_ind).astype(int)

def shuffle_time_series(r):
    n, m = r.shape
    mid = int(m/2)    
    # Create a stack of baseline and post_stim timeseries
    baselines = r[:,0:mid]
    post_stim_part = r[:,mid:None]
    half_time_series_stack = np.concatenate((baselines,post_stim_part),axis=0)
    n_half_rows = half_time_series_stack.shape[0]
    # Shuffle stack rows
    rand_row_ind = np.random.permutation(n_half_rows)
    shuffled_stack = half_time_series_stack[rand_row_ind,:]
    # Regroup into full time series   
    shuff_baselines = shuffled_stack[0:n,:]
    shuff_post_stim = shuffled_stack[n:None,:]
    shuffled_r = np.concatenate((shuff_baselines,shuff_post_stim),
                                       axis=1)    
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
      stat_test - string; should be one of 't'(t-test) or 'ranksum'
    Ouputs:
          stats - 1d numpy array of floats; stat values for each time point
    """            
    n, m = r.shape
    # Step 1: Find mean of baseline for each time series
    baseline_means = []
    for i in range(n):
        baseline_means.append(np.mean(r[i,bin_cen_t<0]))
    # Step 2: Perform statistical test at each time point
    stats = [] # Column vector  
    match stat_test:
        case 't': # t-test
            for iTime in range(m):               
                tval,_ = sps.ttest_rel(r[:,iTime],baseline_means)
                stats.append(tval)
        case 'ranksum': # ranksum test           
            for iTime in range(m):               
                rk_val,_ = sps.ranksums(r[:,iTime],baseline_means)
                stats.append(rk_val)
        case _:
            raise ValueError("stat_test must be 't' or 'ranksum'")
    return np.array(stats)

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
    # Step-1: Get the test statistic value for each time point
    stats = get_stats_for_timeseries(bin_cen_t,r,stat_test) # output: 1d np array
    # Step-2: Find clusters that exceed threshold
    alpha = 0.05/2 # alpha - Type-1 error threshold, two-sided   
    match stat_test:
        case 't': # t-test
            df = r.shape[0]-1 # degrees of freedom
            cval_pos = sps.t.ppf(1-alpha,df)# critical value upper
            cval_neg = sps.t.ppf(alpha,df)# critical value lower
        case 'ranksum':
            # Todo: find critical values
            cval_pos = []
            cval_neg = []
            raise Exception("ranksum test is not implemented yet")
        case _:
            raise ValueError('Only "t" or "ranksum" is allowed')
    # Because we will do a two-sided test, we will cluster positive and negative
    # modulations separately
    pos_ind = stats > cval_pos
    neg_ind = stats < cval_neg    
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
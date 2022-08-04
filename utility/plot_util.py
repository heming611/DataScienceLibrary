import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import calendar
import datetime
from scipy.stats import norm
import pyarrow.parquet as pq
import pyarrow as pa
from time import time
from scipy.stats import norm, ttest_ind
from statsmodels.distributions.empirical_distribution import ECDF

def ecdf_compare(df_group1, df_group2, group1_name, group2_name, metrics, save_path, colors=['#1f77b4','#ff7f0e']):
    '''
    functionality: graph empirical distributions
    '''

    l = len(metrics)
    if l==2:
        plt.figure(figsize=(15,5))
    if l==4:
        plt.figure(figsize=(15,10))

    for i, metric in enumerate(metrics):
        if l==2:
            plt.subplot(1,2,i+1)
        if l==4:
            plt.subplot(2,2,i+1)

        ecdf1 = ECDF(df_group1[metric])
        ecdf2 = ECDF(df_group2[metric])

        t = np.arange(-0.1, 1.2, 0.01)
        ecdf1_values = ecdf1(t)
        ecdf2_values = ecdf2(t)

        plt.plot(t, ecdf1_values, label=group1_name, linewidth=2.0, color=colors[0])
        plt.plot(t, ecdf2_values, label=group2_name, linewidth=2.0, color=colors[1])
        plt.title("empirical cdf for "+metric)
        plt.xlabel('t')
        plt.ylim(0,1.05)
        plt.legend()
  
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return

def hist_compare(df_group1, df_group2, group1_name, group2_name, metrics, save_path, colors=['#1f77b4','#ff7f0e']):
    '''
    functionality: graph histogram for different groups
    '''
    
    l = len(metrics)
    if l==2:
        plt.figure(figsize=(15,5))
    if l==4:
        plt.figure(figsize=(15,10))

    for i, metric in enumerate(metrics):
        if l==2:
            plt.subplot(1,2,i+1)
        if l==4:
            plt.subplot(2,2,i+1)

        weights = [np.ones_like(df_group1[metric])/float(len(df_group1[metric])), 
                   np.ones_like(df_group2[metric])/float(len(df_group2[metric]))]

        plt.hist([df_group1[metric], df_group2[metric]],
                 bins = 20, color = colors, 
                 label = [group1_name, group2_name],
                 weights = weights)
        plt.legend()
        plt.xlabel(metric)
        plt.ylabel('normalized frequency')

        if metric in ("full-conversion","billing-conversion"):
            plt.title('Histogram for ' + metric)
        else:
            plt.title('Histogram for Avg(' + metric + ' rate)')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
   
    return



import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep
from iqlclient import IQLClient
from iqlclient.exceptions import (IQLNoShards, IQLServerError)
from dateutil import parser
from pyhive import hive
import logging
import pwd
import pandas as pd
import requests_kerberos
import pyhive
from pyhive import presto
from multiprocessing.dummy import Pool
from typing import List


class RankingMetrics:
    
    def DCG_at_p(self, relevance, p):
        '''
        functinality: given a list of relevance scores for each position, compute DCG score for ranking up to position p
        '''
        DCG_score = 0
        relevance = list(relevance)

        for i, j in enumerate(relevance[:p]):
            if i == 0:
                DCG_score += j
            else:
                DCG_score += j/np.log(i+2)

        return DCG_score


    def IDCG_at_p(self, relevance, p):
        '''
        functionality: given a list of relevance scores for each position, compute the ideal DCG score for ranking up to position p
        '''
        relevance_sorted = sorted(list(relevance), reverse = True)
        #print(sorted_relevance)

        return self.DCG_at_p(relevance_sorted, p)


    def NDCG_at_p(self, relevance, p):
        '''
        functionality: given a list of relevance scores for each position, compute the normalized DCG for ranking up to position p 
        '''
        dcg = self.DCG_at_p(relevance, p)
        #print(dcg)
        idcg = self.IDCG_at_p(relevance, p)
        #print(idcg)
        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg*1.0/idcg
        #print(ndcg)

        return ndcg


    def precision_at_p(self, relevance, p):
        '''
        functionality: given a list of relevance scores for each position, compute precision at p >= 1 for a ranking
        '''
        return round(np.sum(relevance[:p])*1.0/p, 4)


    def MAP_at_p(self, relevance, p):
        '''
        functionality: given a list of relevance scores for each position, compute mean average precision at p for a ranking
        relevance: a list of relevance score for all rankded documents
        '''
        mean_ave_precision = 0
        relevance = np.array(relevance[:p])
        #print(relevance)
        num_of_relevant_doc = np.sum(relevance > 0)
        #print("number of relevant docs", num_of_relevant_doc)

        if num_of_relevant_doc == 0:
            return 0
        else:
            for i, j in enumerate(relevance):
                #print(i, j)
                if j > 0:
                    mean_ave_precision += precision_at_p(relevance[:(i+1)], i+1)
                    #print("x",mean_ave_precision)
            return round(mean_ave_precision*1.0/num_of_relevant_doc, 4)
        
        
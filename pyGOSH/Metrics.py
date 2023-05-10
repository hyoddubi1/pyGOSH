# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:49:04 2022

@author: user
"""

#Metaheuristic otimization algorithms
# Hyoseb Noh
# Update date : 2022 06 22
#%%
import numpy as np
#%%
def R2(y_obs,y_pred):
    y_mean = np.mean(y_pred)
    r2 = 1 - np.sum((y_pred - y_obs)**2)/np.sum((y_obs-y_mean)**2)
    return r2

def adjusted_R2(y_obs,y_pred,k):
    r2 = R2(y_obs,y_pred)
    n = len(y_obs)
    C = (n-1)/(n-k-1)
    T2 = 1 - r2
    return 1 - C*T2

def MSE(y_obs,y_pred):
    n = len(y_obs)
    return np.sum((y_pred - y_obs)**2)/n

def RMSE(y_obs,y_pred):
    n = len(y_obs)
    return np.sqrt(np.sum((y_pred - y_obs)**2))/n

def neg_R2(y_obs,y_pred):
    return -R2(y_obs,y_pred)

def neg_adjusted_R2(y_obs,y_pred):
    return -adjusted_R2(y_obs,y_pred)

def MAE(y_obs,y_pred):
    n = len(y_obs)
    return np.sum(np.abs(y_pred - y_obs))/n


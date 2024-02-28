# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:49:04 2022

@author: user
"""

#Metaheuristic otimization algorithms
# Hyoseb Noh
# Update date : 2022 06 22
#%%
#from statsmodels.stats.outliers_influence import variance_inflation_factor
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

def variance_inflation_factor(exog, exog_idx):
    """
    It has been copied from 'statsmodel' library
    Variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : {ndarray, DataFrame}
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    float
        variance inflation factor

    Notes
    -----
    This function does not save the auxiliary regression.

    See Also
    --------
    xxx : class for regression diagnostics  TODO: does not exist yet

    References
    ----------
    https://en.wikipedia.org/wiki/Variance_inflation_factor
    """
    k_vars = exog.shape[1]
    exog = np.asarray(exog)
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


def VIF(X):
    return np.array([variance_inflation_factor(X,i) for i in range(X.shape[1])])


def LinearRankCriterion(estimator):
    svs = estimator.support_vectors_
    duals = estimator.dual_coef_
    duals = duals.transpose()
    c = np.sum(svs*duals,axis = 0)
    return c**2
    

def rbfRankCriterion(estimator):
    c = []
    svs = estimator.support_vectors_
    duals = estimator.dual_coef_
    duals = duals.transpose()
    dualmatmul = np.matmul(duals,duals.transpose())
    N,Nvar = svs.shape
    Kmat = np.zeros([N,N])
    Kmat_p = np.zeros([N,N])
    xi3d = svs.reshape([N,1,Nvar]).repeat(N,axis = 1)
    xj3d = svs.reshape([1,N,Nvar]).repeat(N,axis = 0)
    Kmat = np.sum((xi3d - xj3d)**2,axis = 2)
    for p in range(Nvar):

        X_p = np.delete(svs,p,axis = 1)

        xi3d_p = X_p.reshape([N,1,Nvar-1]).repeat(N,axis = 1)
        xj3d_p = X_p.reshape([1,N,Nvar-1]).repeat(N,axis = 0)
        Kmat_p = np.sum((xi3d_p - xj3d_p)**2,axis = 2)
    
        T1 = dualmatmul*Kmat
        T2 = dualmatmul*Kmat_p
        
        D = np.sum(T1) - np.sum(T2)
        c.append(D)
        
    c = np.array(c)
    return c**2
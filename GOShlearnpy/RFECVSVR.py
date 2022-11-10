# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:06:06 2022

@author: user
"""

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold

#import pickle
#import joblib
import numpy as np
from sklearn.metrics import r2_score
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import time
import numbers
from sklearn.base import clone

from tqdm import tqdm


#%%

def _fit_and_score(
    estimator,
    X,
    y,
    train_index,
    test_index,
    PreProcessory,
    PostProcessor,
    cv,
    return_train_score=None,
):
    result = {}
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if PreProcessory:
        y_train = PreProcessory.fit_transform(y_train)
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    
    if PostProcessor:
        y_pred = PostProcessor.fit_transform(y_pred)
    y_pred = np.nan_to_num(y_pred)
    result["test_scores"] = r2_score(y_test,y_pred)
    
    if return_train_score:
        if PreProcessory:
            y_train = PreProcessory.fit_transform(y_train)
        y_pred = estimator.predict(X_test)
        if PostProcessor:
            y_pred = PostProcessor.fit_transform(y_pred)
        result["train_scores"] = r2_score(y_train,y_pred)

    return result

def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray

    The aggregated output of _aggregate_score_dicts will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}

    Parameters
    ----------

    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {
        key: np.asarray([score[key] for score in scores])
        if isinstance(scores[0][key], numbers.Number)
        else [score[key] for score in scores]
        for key in scores[0]
    }

def power10(x):
#    return np.power(10,x)
    return 10**x

def PrePostProcessor(yScale= None):
    

    if yScale == 'log':
        PreProcessor = FunctionTransformer(np.log)
        PostProcessor = FunctionTransformer(np.exp)
    elif yScale == 'exp':
        PreProcessor = FunctionTransformer(np.exp)
        PostProcessor = FunctionTransformer(np.log)
    elif yScale == 'log10':
        PreProcessor = FunctionTransformer(np.log10)
        PostProcessor = FunctionTransformer(power10)
    elif yScale == 'power10':
        PreProcessor = FunctionTransformer(power10)
        PostProcessor = FunctionTransformer(np.log10)
    else:
        PreProcessor = None
        PostProcessor = None
        
    return PreProcessor, PostProcessor

def ParCVR2(
        tempestimator,
        X,
        y,
        n_features = None,
        n_jobs = None,
        PreProcessor=None,
        PostProcessor=None,
        cv=KFold(n_splits=5, random_state=42,  shuffle=True),
        return_train_score = None,
        pre_dispatch = '2*n_jobs',
    ):
    

    if n_jobs == None:
        results =[
        _fit_and_score(
            tempestimator,
            X,
            y,
            train_index,
            test_index,
            PreProcessor,
            PostProcessor,
            cv,
            return_train_score = return_train_score,
        )
        for train_index, test_index in cv.split(X)
        ]
    else:
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch = pre_dispatch)
        results = parallel(
            delayed(_fit_and_score)(
                tempestimator,
                X,
                y,
                train_index,
                test_index,
                PreProcessor,
                PostProcessor,
                cv,
                return_train_score = return_train_score,
            )
            for train_index, test_index in cv.split(X)
        )
    results = _aggregate_score_dicts(results)
#    results['mean_test_scores'] = np.mean(results['test_scores'])
    

    return results

class RFE(object):
    def __init__(self,estimator=None,min_features_to_select = 1,verbose = False,cv = None,scoring = 'r2',n_jobs = None,
                 return_train_score=False,yScale = None,transform_exp_X= None,transform_exp_y = None, pre_dispatch = '2*n_jobs'):
        if estimator == None:
            from sklearn.svm import SVR
            estimator = SVR()
        self.estimator = estimator
        if cv == None:
            self.cv = None
        else:
            self.cv = cv
            self.n_jobs = n_jobs
        self.scoring = scoring
        self.return_train_score = return_train_score
        self.min_features_to_select = min_features_to_select
        self.PreProcessor, self.PostProcessor = PrePostProcessor(yScale= yScale)
        self.yScale = yScale
        self.transform_exp_X = transform_exp_X
        self.transform_exp_y = transform_exp_y
        self.pre_dispatch = pre_dispatch
        
        
        
    def LinearRankCriterion(self,estimator):
        svs = estimator.support_vectors_
        duals = estimator.dual_coef_
        duals = duals.transpose()
        c = np.sum(svs*duals,axis = 0)
        return c**2
        
    
    def rbfRankCriterion(self,estimator):
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
    
    def SVRRankCriterion(self,X,y,estimator):
        estimator.fit(X,y)
        if str(type(self.estimator))[-5:-2] == 'SVR':
            if estimator.kernel =='linear':
                c = self.LinearRankCriterion(estimator)
            elif estimator.kernel == 'rbf':
                c = self.rbfRankCriterion(estimator)
        elif str(type(self.estimator))[-10:-2] == 'Pipeline':
            if estimator['regressor'].kernel =='linear':
                c = self.LinearRankCriterion(estimator['regressor'])
            elif estimator['regressor'].kernel == 'rbf':
                c = self.rbfRankCriterion(estimator['regressor'])
            
        return c
    
    def fit(self,X,y):
        nVars = X.shape[1]
        Vars = np.arange(0,nVars)
        InvNumbs = np.arange(self.min_features_to_select,nVars)[::-1]
        survived_vars = []
        eliminated_vars = []
        ranking_coefficeints = []
        
        if self.transform_exp_X:
            X = np.exp(X)
        if self.transform_exp_y:
            y = np.exp(y)
            

        
        if self.cv != None:
            test_scores = []
            mean_test_scores = []
            if self.return_train_score:
                train_scores = []
                mean_train_scores = []
                
        for i in tqdm(InvNumbs):
            time.sleep(0.0001)
            tempX = X[:,Vars]
            if self.cv != None:
#                score = cross_validate(self.estimator, tempX, y, cv=self.cv, n_jobs = self.n_jobs,
#                        scoring=self.scoring,
#                        return_train_score=self.return_train_score)
                score = ParCVR2(
                        clone(self.estimator),
                        tempX,
                        y,
                        n_features = None,
                        n_jobs = self.n_jobs,
                        PreProcessor=self.PreProcessor,
                        PostProcessor=self.PostProcessor,
                        cv=self.cv,
                        return_train_score = self.return_train_score,
                        pre_dispatch = self.pre_dispatch,
                    )
#                results = _aggregate_score_dicts(results)
                test_scores.append(score['test_scores'])
                mean_test_scores.append(np.mean(score['test_scores']))
#                mean_test_scores.append(score['mean_test_score'])
                
                if self.return_train_score:
                    train_scores.append(score['train_scores'])
                    mean_train_scores.append(np.mean(score['train_scores']))
            
            c = self.SVRRankCriterion(tempX,y,self.estimator)
            dead = c.argmin()
            dead = Vars[dead]
            Vars = Vars[Vars!=dead]
            survived_vars.append(Vars)
            eliminated_vars.append(dead)
            ranking_coefficeints.append(c)
        
        eliminated_vars.append(Vars[0]) 
        survived_vars = survived_vars[::-1]
        importance_orders = np.array(eliminated_vars[::-1])
        feature_ranking = np.arange(nVars)
        feature_ranking[importance_orders] = np.arange(nVars)
        feature_ranking = feature_ranking + 1
        self.rfe_results = {'survived_vars':survived_vars,'feature importance orders':importance_orders,
                            'feature importance ranking':feature_ranking}

        if self.cv != None:
            tempX = X[:,Vars]  
#            score = cross_validate(self.estimator, tempX, y, cv=self.cv, n_jobs = self.n_jobs,
#                        scoring=self.scoring,
#                        return_train_score=self.return_train_score)
            score = ParCVR2(
                        clone(self.estimator),
                        tempX,
                        y,
                        n_features = None,
                        n_jobs = self.n_jobs,
                        PreProcessor=self.PreProcessor,
                        PostProcessor=self.PostProcessor,
                        cv=self.cv,
                        return_train_score = self.return_train_score,
                    )
            test_scores.append(score['test_scores'])
            mean_test_scores.append(np.mean(score['test_scores']))
            self.cv_results = {'test_scores':np.array(test_scores[::-1]),'mean_test_scores':np.array(mean_test_scores[::-1])}
            if self.return_train_score:
                train_scores.append(np.mean(score['test_scores']))
                self.cv_results['train_scores'] = np.array(train_scores[::-1])
                self.cv_results['mean_train_scores'] = np.array(mean_train_scores[::-1])
                
    def predict(self,X):
        
        y_pred = self.estimator.predict(X)
        if self.yScale:
            if self.PostProcessor:
                y_pred = self.PostProcessor.fit_transform(y_pred)
        
        return y_pred

class GRIDRFE(object):
    def __init__(self,rfe_object,parameters = None):
        self.rfe = rfe_object
        if parameters == None:
            C_exp = [-3,-2,-1,0,1,2,3]
            Cs = [2**i for i in C_exp]
            epsilons = [10**-3,10**-2,10**-1,0]
            
            self.parameters = {'C':Cs,'epsilon':epsilons,'gamma':['scale'] }
        else:
            self.parameters = {'C':[1],'epsilon':[0.1],'gamma':['scale'] }
            if 'C' in parameters:
                self.parameters['C'] = parameters['C']
            if 'epsilon' in parameters:
                self.parameters['epsilon'] = parameters['epsilon']
            if 'gamma' in parameters:
                self.parameters['gamma'] = parameters['gamma']
        
   
    def fit_single_estimator(self,X,y):
        grid_rankings = []
        Parameters = []
        if self.rfe.cv != None:
            mean_test_scores = []
        for C in self.parameters['C']:
            self.rfe.estimator.C = C
            for epsilon in self.parameters['epsilon']:
                self.rfe.estimator.epsilon = epsilon
                for gamma in self.parameters['gamma']:
                    self.rfe.estimator.gamma = gamma
                    
                    self.rfe.fit(X,y)
                    if self.rfe.cv != None:
                        mean_test_scores.append(self.rfe.cv_results['mean_test_scores'])
                    
                    grid_rankings.append(self.rfe.rfe_results['feature importance ranking'])
                    Parameters.append([C,epsilon,gamma])
        grid_rankings = np.array(grid_rankings)
        ranking_sum = np.sum(grid_rankings,axis = 0)
        
        self.gridrfe_results = {'hyperparameters':Parameters,
                                'grid_rankings':grid_rankings,
                                'ranking_sum':ranking_sum}
        if self.rfe.cv != None:
            self.gridrfe_results['mean_test_scores'] = np.array(mean_test_scores)
            self.gridrfe_results['average_mean_test_scores'] = np.mean(self.gridrfe_results['mean_test_scores'], axis = 0)
            
            
    def fit_pipeline(self,X,y):
        grid_rankings = []
        Parameters = []
        
        if self.rfe.cv != None:
            mean_test_scores = []
        for C in self.parameters['C']:
            self.rfe.estimator['regressor'].C = C
            for epsilon in self.parameters['epsilon']:
                self.rfe.estimator['regressor'].epsilon = epsilon
                for gamma in self.parameters['gamma']:
                    self.rfe.estimator['regressor'].gamma = gamma
                    
                    self.rfe.fit(X,y)
                    if self.rfe.cv != None:
                        mean_test_scores.append(self.rfe.cv_results['mean_test_scores'])
                    
                    grid_rankings.append(self.rfe.rfe_results['feature importance ranking'])
                    Parameters.append([C,epsilon,gamma])
        grid_rankings = np.array(grid_rankings)
        ranking_sum = np.sum(grid_rankings,axis = 0)
        
        self.gridrfe_results = {'hyperparameters':Parameters,
                                'grid_rankings':grid_rankings,
                                'ranking_sum':ranking_sum}
        if self.rfe.cv != None:
            self.gridrfe_results['mean_test_scores'] = np.array(mean_test_scores)
            self.gridrfe_results['average_mean_test_scores'] = np.mean(self.gridrfe_results['mean_test_scores'], axis = 0)
            self.gridrfe_results['average_max_test_scores'] = np.max(self.gridrfe_results['mean_test_scores'], axis = 0)
    
    def fit(self,X,y):
        if str(type(self.rfe.estimator))[-5:-2] == 'SVR':
            self.fit_single_estimator(X,y)
        elif str(type(self.rfe.estimator))[-10:-2] == 'Pipeline':
            self.fit_pipeline(X,y)              
            
    def find_best(self,n_features=None):
        
        test_score_mat = self.gridrfe_results['mean_test_scores']
         
        
        if n_features == None:
            best_score_loc = np.unravel_index(test_score_mat.argmax(), test_score_mat.shape)#(Parameter grid, n_features)
            n_features = best_score_loc[1]+1
        else:
            best_score_loc = test_score_mat[:,n_features-1].argmax() 
            best_score_loc = [best_score_loc,n_features-1]
        
        bestParameters = self.gridrfe_results['hyperparameters'][best_score_loc[0]]
        bestRanking = self.gridrfe_results['grid_rankings'][best_score_loc[0]]
        Support = self.gridrfe_results['grid_rankings'][best_score_loc[0]] <= n_features
        bestScore = self.gridrfe_results['mean_test_scores'][best_score_loc[0],best_score_loc[1]]
        
        self.best_gridrfe_result = {'best_score_location':best_score_loc,
                                    'hyperparameters':bestParameters,
                                    'n_features':n_features,
                                    'ranking':bestRanking,
                                    'support':Support,
                                    'score':bestScore}
        return self.best_gridrfe_result
    
    def find_best_fit(self,X,y,n_features=None):
        
        self.find_best(n_features = n_features)
        
        Xtrain = X[:,self.best_gridrfe_result['support']]
        
        if self.rfe.yScale:
            if self.rfe.PreProcessor:
                y = self.rfe.PreProcessor.fit_transform(y)
            
        if str(type(self.rfe.estimator))[-5:-2] == 'SVR':
            self.rfe.estimator.C = self.best_gridrfe_result['hyperparameters'][0]
            self.rfe.estimator.epsilon = self.best_gridrfe_result['hyperparameters'][1]
            self.rfe.estimator.gamma = self.best_gridrfe_result['hyperparameters'][2]
            self.rfe.estimator.fit(Xtrain,y)
        elif str(type(self.rfe.estimator))[-10:-2] == 'Pipeline':
            self.rfe.estimator['regressor'].C = self.best_gridrfe_result['hyperparameters'][0]
            self.rfe.estimator['regressor'].epsilon = self.best_gridrfe_result['hyperparameters'][1]
            self.rfe.estimator['regressor'].gamma = self.best_gridrfe_result['hyperparameters'][2]
            self.rfe.estimator.fit(Xtrain,y)
        
        
        return self.best_gridrfe_result
    
    def predict(self,X):
#        y = self.rfe.estimator.predict(X)
        y_pred = self.rfe.estimator.predict(X)
        if self.rfe.yScale:
            if self.rfe.PostProcessor:
                y_pred = self.rfe.PostProcessor.fit_transform(y_pred)
        
        return y_pred

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
from joblib import Parallel, delayed, dump, load
from joblib import wrap_non_picklable_objects
import time
import numbers
from sklearn.base import clone

from tqdm import tqdm

from . import GlobalOptimization as go
from . import Metrics 



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
        self.verbose = verbose
        
        
        
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
        survived_vars.append(Vars)
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
                
        for i in tqdm(InvNumbs, leave= self.verbose):
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
    
    def export_estimator(self,
               model_path='rfe_svr.pkl'
               ):
        
        dump(self.estimator, model_path) 
        
        return
        
    def import_estimator(self,
               model_path='rfe_svr.pkl'
               ):
        
        self.estimator = load(model_path) 
        
        return
        

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

    def export_estimator(self,
               model_path='gridrfe_svr.pkl'
               ):
        
        dump(self.estimator, model_path) 
        
        return
        
    def import_estimator(self,
               model_path='gridrfe_svr.pkl'
               ):
        
        self.estimator = load(model_path) 
        
        return
        

class GORFE(object):
    def __init__(self,estimator=None,
                 rfe=None,
                 optimizer=None,
                 opt_flag = [1,1,1],
                 verbose = True,
                 X_log_flag = [False,True,True]):
        if estimator is None:
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVR
            from sklearn.pipeline import Pipeline
            svr = SVR(kernel='rbf')
            scaler = StandardScaler()
            estimator = Pipeline([('scaler',scaler),
                             ('regressor',svr)]) 
        
        self.rfe = rfe
        if self.rfe is None:
   
            k_fold =  KFold(n_splits=3, random_state=42,  shuffle=True)
            self.rfe = RFE(estimator,cv = k_fold, n_jobs = 1)
            
        self.optimizer = optimizer # optimizer's parameter space dimension has to be equalt to opt_flag
        if self.optimizer is None:
            self.optimizer = go.Optimizer(
                    lb = [10**-9,10**-9,10**-9],
                    ub = [10000,10000,10000],
                    algorithm = 'MCCE',
                    stop_step = 50,
                    stop_span = 10**-7,
                    stop_obj_percent = 0.1,
                    stop_fcal = 10000,
                    dimension_restore = True,
                    n_complex = 4,
                    n_complex_size = 8,
                    iseed = None,
                    iniflg = None,
                    pop_init_method = 'LHS',
                    init_pop = None,
                    verbose = verbose,
                    algorithm_params = 'default',
                    n_jobs = 1,
                    pre_dispatch = '2*n_jobs',
                    obj_eval = 'serial'
                    ) 
        
        # self.opt_flag = opt_flag
        
        self.X_log_flag = X_log_flag
        
        # self.optimizer.lb = self.optimizer.lb
        
        self.optimizer.lb[self.X_log_flag] = np.log10(self.optimizer.lb[self.X_log_flag])
        self.optimizer.ub[self.X_log_flag] = np.log10(self.optimizer.ub[self.X_log_flag])
        
        # print(self.optimizer.ub, self.optimizer.lb)
        
    def obj_func_rfe_pipeline(self,x):
        
        # print('before',x)
        x2 = x.copy()
        x2[self.X_log_flag] =10**x2[self.X_log_flag]
        # print('after',x)
        rfe = self.rfe
                
        rfe.estimator['regressor'].C = x2[0]
        rfe.estimator['regressor'].epsilon = x2[1]
        rfe.estimator['regressor'].gamma = x2[2]
        
        rfe.fit(self.X,self.y)
        
        maxscore = np.max(self.rfe.cv_results['mean_test_scores'])
        # print(self.rfe.cv_results['mean_test_scores'])
        
        f = 1 - maxscore
        
        
        return f
    
    def obj_func_rfe_single_estimator(self,x):
        
        x2 = x.copy()
        x2[self.X_log_flag] =10**x2[self.X_log_flag]
        
        rfe = self.rfe
        rfe.estimator.C = x2[0]
        rfe.estimator.epsilon = x2[1]
        rfe.estimator.gamma = x2[2]
        
        rfe.fit(self.X,self.y)
        
        maxscore = np.max(self.rfe.cv_results['mean_test_scores'])
        
        f = 1 - maxscore
        
        return f
   
    def fit(self,X,y):
        self.X = X
        self.y = y
        if str(type(self.rfe.estimator))[-5:-2] == 'SVR':
            # self.fit_single_estimator(X,y)
            self.optimizer.obj_func = self.obj_func_rfe_single_estimator
            self.bestX, self.bestf = self.optimizer.evolve()
            
            self.bestX[self.X_log_flag] = 10**self.bestX[self.X_log_flag]
            self.rfe.estimator.C = self.bestX[0]
            self.rfe.estimator.epsilon = self.bestX[1]
            self.rfe.estimator.gamma = self.bestX[2]
            
        elif str(type(self.rfe.estimator))[-10:-2] == 'Pipeline':
            # self.fit_pipeline(X,y)              
            self.optimizer.obj_func = self.obj_func_rfe_pipeline
            self.bestX, self.bestf = self.optimizer.evolve()
            
            self.bestX[self.X_log_flag] = 10**self.bestX[self.X_log_flag]
            self.rfe.estimator['regressor'].C = self.bestX[0]
            self.rfe.estimator['regressor'].epsilon = self.bestX[1]
            self.rfe.estimator['regressor'].gamma = self.bestX[2]
        
                    
        print('\r\n Fitting the rfe module with the best parameter combination...')
        
        self.rfe.fit(self.X,self.y)
        
        n_features = np.argmax(self.rfe.cv_results['mean_test_scores']) + 1
        Support = self.rfe.rfe_results['survived_vars'][n_features-1]
        bestRanking = self.rfe.rfe_results['feature importance ranking']
        bestScore = np.max(self.rfe.cv_results['mean_test_scores'])
        self.best_gorfe_result = {'hyperparameters': self.bestX,
                                  'n_features':n_features,
                                  'ranking':bestRanking,
                                  'support':Support,
                                  'score':bestScore}
        
        print('\n cv results: \n',self.rfe.cv_results)
        print('\n rfe results: \n',self.rfe.rfe_results)
        print('\n gorfe results: \n',self.best_gorfe_result)
        
        Xtrain = X[:,Support]
        
        if self.rfe.yScale:
            if self.rfe.PreProcessor:
                y = self.rfe.PreProcessor.fit_transform(y)
            
        if str(type(self.rfe.estimator))[-5:-2] == 'SVR':
            self.rfe.estimator.C = self.best_gorfe_result['hyperparameters'][0]
            self.rfe.estimator.epsilon = self.best_gorfe_result['hyperparameters'][1]
            self.rfe.estimator.gamma = self.best_gorfe_result['hyperparameters'][2]
            self.rfe.estimator.fit(Xtrain,y)
        elif str(type(self.rfe.estimator))[-10:-2] == 'Pipeline':
            self.rfe.estimator['regressor'].C = self.best_gorfe_result['hyperparameters'][0]
            self.rfe.estimator['regressor'].epsilon = self.best_gorfe_result['hyperparameters'][1]
            self.rfe.estimator['regressor'].gamma = self.best_gorfe_result['hyperparameters'][2]
            self.rfe.estimator.fit(Xtrain,y)
        
        return self.best_gorfe_result
        
    def fit_best(self,X,y):    
        Xtrain = X[:,self.best_gorfe_result['support']]
        
        if self.rfe.yScale:
            if self.rfe.PreProcessor:
                y = self.rfe.PreProcessor.fit_transform(y)
            
        if str(type(self.rfe.estimator))[-5:-2] == 'SVR':
            self.rfe.estimator.C = self.best_gorfe_result['hyperparameters'][0]
            self.rfe.estimator.epsilon = self.best_gorfe_result['hyperparameters'][1]
            self.rfe.estimator.gamma = self.best_gorfe_result['hyperparameters'][2]
            self.rfe.estimator.fit(Xtrain,y)
        elif str(type(self.rfe.estimator))[-10:-2] == 'Pipeline':
            self.rfe.estimator['regressor'].C = self.best_gorfe_result['hyperparameters'][0]
            self.rfe.estimator['regressor'].epsilon = self.best_gorfe_result['hyperparameters'][1]
            self.rfe.estimator['regressor'].gamma = self.best_gorfe_result['hyperparameters'][2]
            self.rfe.estimator.fit(Xtrain,y)
        
    def predict(self,X):
        
        if X.shape[1] != self.best_gorfe_result['n_features']:
            print('Since the shape of the input matrix is not equal to the best result,',
                  ' it regards the input as the best X and converts the size corresponding to the "support"')
            X = X[:,self.best_gorfe_result['support']]
        
        y_pred = self.rfe.estimator.predict(X)
        if self.rfe.yScale:
            if self.rfe.PostProcessor:
                y_pred = self.rfe.PostProcessor.fit_transform(y_pred)
        
        return y_pred
    
    def export_estimator(self,
               model_path='gorfe_svr.pkl'
               ):
        
        dump(self.estimator, model_path) 
        
        return
        
    def import_estimator(self,
               model_path='gorfe_svr.pkl'
               ):
        
        self.estimator = load(model_path) 
        
        return
        
        


class MOSGOSVR(object): #Model selection with global optimization for SVR (MoSGO-SVR)
    def __init__(self,X=None,
                 estimator=None,
                 cv = None,
                 optimizer=None,
                 verbose = True,
                 X_log_flag = [False,True,True],
                 yScale = None,
                 n_jobs= 1,
                 fix_variables = None,
                 int_mode = 'eval',
                 lamb = 0,
                 n_runs = 1):
        
        self.int_mode = int_mode # 'eval', 'opt'
        
        if cv == None:
            self.cv = KFold(n_splits=3, random_state=42,  shuffle=True)
            
        else:
            self.cv = cv
            self.n_jobs = n_jobs
        
        if estimator is None:
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVR
            from sklearn.pipeline import Pipeline
            svr = SVR(kernel='rbf')
            scaler = StandardScaler()
            self.estimator = Pipeline([('scaler',scaler),
                             ('regressor',svr)]) 
        else:
            self.estimator = estimator
            
        self.optimizer = optimizer # optimizer's parameter space dimension has to be equalt to opt_flag
        if self.optimizer is None:
            self.optimizer = go.Optimizer(
                    lb = [10**-9,10**-9,10**-9],
                    ub = [10000,10000,10000],
                    algorithm = 'MCCE',
                    stop_step = 50,
                    stop_span = 10**-7,
                    stop_obj_percent = 0.1,
                    stop_fcal = 10000,
                    dimension_restore = True,
                    n_complex = 4,
                    n_complex_size = 8,
                    iseed = None,
                    iniflg = None,
                    pop_init_method = 'LHS',
                    init_pop = None,
                    verbose = verbose,
                    algorithm_params = 'default',
                    n_jobs = 1,
                    pre_dispatch = '2*n_jobs',
                    obj_eval = 'serial'                    
                    ) 
        
        self.lamb = lamb
        # self.opt_flag = opt_flag
        
        self.X_log_flag = X_log_flag
        
        # self.optimizer.lb = self.optimizer.lb
        
        self.optimizer.lb[self.X_log_flag] = np.log10(self.optimizer.lb[self.X_log_flag])
        self.optimizer.ub[self.X_log_flag] = np.log10(self.optimizer.ub[self.X_log_flag])
        self.n_runs = n_runs
                
        if X is not None:
            self.X = X.copy()
            self.nx, self.nv = X.shape
            self.optimizer.ub =np.append(np.ones(self.nv),self.optimizer.ub)
            self.optimizer.lb =np.append(np.zeros(self.nv),self.optimizer.lb)
            int_program_v = np.ones(self.nv,dtype=bool)
            
            if self.int_mode == 'opt':
                self.optimizer.int_program = np.append(int_program_v,[False,False,False])
            else:
                self.optimizer.int_program = False

        else:
            self.nv = None

            
        # print(self.X_log_flag)

        self.yScale = yScale
        self.PreProcessor, self.PostProcessor = PrePostProcessor(yScale= self.yScale)
        # print(self.optimizer.ub, self.optimizer.lb)
        self.fix_variables = fix_variables
        # self.X_log_flag
        
    def obj_func_rfe_pipeline(self,x):
        
        # print('before',x)
        x2 = x.copy()
        x2[self.X_log_flag] =10**x2[self.X_log_flag]
        # print(np.sum(x2[:-3]))
        # print('after',x)
        estimator = clone(self.estimator)
                
        estimator['regressor'].C = x2[-3]
        estimator['regressor'].epsilon = x2[-2]
        estimator['regressor'].gamma = x2[-1]
        
        if np.sum(x2[:-3]) == 0:
            f = 10**9
            return f
        else:
            # if self.int_mode == 'opt':
            #     tempX = self.X[:,x2[:-3]==1].copy()
            # else:
            #     tempX = self.X[:,np.round(x2[:-3]).astype('int')].copy()
            
            if self.int_mode == 'opt':
                support = x2[:-3]==1 
                
            else:
                support = np.round(x2[:-3]).astype('bool')
            tempX = self.X[:,support].copy()
            
            score = ParCVR2(
                        estimator,
                        tempX,
                        self.y,
                        n_features = None,
                        n_jobs = self.n_jobs,
                        PreProcessor=self.PreProcessor,
                        PostProcessor=self.PostProcessor,
                        cv=self.cv,
                        return_train_score = False,
                    )
      
            meanscore = np.mean(score['test_scores'])
            
            f = 1 - meanscore
            
            # nv = np.sum(x2[:-3])
            # lamb = 0.2
            # f = (nv/self.nv)*lamb- meanscore
            # f = (nv/self.nv)*self.lamb + f 
            # print(support)
            # print(tempX)
            if np.sum(support)>1:
                vifs = Metrics.VIF(tempX)
            else:
                vifs = 0
            f = np.sum(vifs>5) + f
        
        return f
    
    def obj_func_rfe_single_estimator(self,x):
        
        x2 = x.copy()
        x2[self.X_log_flag] =10**x2[self.X_log_flag]
        # print('after',x)
        estimator = clone(self.estimator)
                
        estimator.C = x2[-3]
        estimator.epsilon = x2[-2]
        estimator.gamma = x2[-1]
        
        
        
        if np.sum(x2[:-3]) == 0:
            f = 10**9
        else:
        
            # tempX = self.X[:,x2[:-3]==1].copy()
            if self.int_mode == 'opt':
                support = x2[:-3]==1 
                
            else:
                support = np.round(x2[:-3]).astype('bool')
            tempX = self.X[:,support].copy()
            score = ParCVR2(
                        estimator,
                        tempX,
                        self.y,
                        n_features = None,
                        n_jobs = self.n_jobs,
                        PreProcessor=self.PreProcessor,
                        PostProcessor=self.PostProcessor,
                        cv=self.cv,
                        return_train_score = self.return_train_score,
                    )
      
            meanscore = np.mean(score['test_scores'])
            
            f = 1 - meanscore
            # nv = np.sum(x2[:-3])
            # f = (nv/self.nv)*self.lamb + f 
            
            # if 
            vifs = Metrics.VIF(tempX)
            
            f = np.sum(vifs>10) + f
            
        return f
    def fit(self,X,y):
        
        self.fit_init_(X,y)
        
        self.Xarray = []
        self.farray = []
        
        for i in range(self.n_runs):
                       
            self.fit_(X,y)
            self.Xarray.append(np.array(self.optimizer.Xs))
            self.farray.append(np.array(self.optimizer.fs))
            
            if self.n_runs > 1:
                print('\r\n %d-th run complete \r\n'%(i+1))
          
        self.Xarray =np.concatenate(self.Xarray)
        self.farray =np.concatenate(self.farray)
        
        return self.best_mosgo_result
        
    def fit_init_(self,X,y):
        
        if self.nv is None:
            self.X = X.copy()
            self.nx, self.nv = X.shape
            int_program_v = np.ones(self.nv,dtype=bool)
            self.optimizer.ub =np.append(np.ones(self.nv),self.optimizer.ub)
            self.optimizer.lb =np.append(np.zeros(self.nv),self.optimizer.lb)
            
            if self.int_mode == 'opt':
                self.optimizer.int_program = np.append(int_program_v,[False,False,False])
            else:
                self.optimizer.int_program = False
            # print(np.append(int_program_v,self.optimizer.int_program))
            if self.fix_variables == True:
                self.fix_variables = np.ones(self.nv,dtype=bool)
            elif self.fix_variables == None:
                self.fix_variables = np.zeros(self.nv,dtype=bool)
                
        for i, flag in enumerate(self.fix_variables):
            if flag==True:
                self.optimizer.lb[i] = 0.99

        self.X_log_flag = np.append(np.zeros(self.nv,dtype=bool),self.X_log_flag)
        
        self.X = X
        self.y = y
        
    def fit_(self,X,y):
        
        # fixing input variables 
        

        if str(type(self.estimator))[-5:-2] == 'SVR':
            # self.fit_single_estimator(X,y)
            self.optimizer.obj_func = self.obj_func_rfe_single_estimator
            self.bestX, self.bestf = self.optimizer.evolve()
            
            self.bestX[self.X_log_flag] = 10**self.bestX[self.X_log_flag]
            self.estimator.C = self.bestX[-3]
            self.estimator.epsilon = self.bestX[-2]
            self.estimator.gamma = self.bestX[-1]
            
        elif str(type(self.estimator))[-10:-2] == 'Pipeline':
            # self.fit_pipeline(X,y)              
            self.optimizer.obj_func = self.obj_func_rfe_pipeline
            self.bestX, self.bestf = self.optimizer.evolve()
            
            self.bestX[self.X_log_flag] = 10**self.bestX[self.X_log_flag]
            self.estimator['regressor'].C = self.bestX[-3]
            self.estimator['regressor'].epsilon = self.bestX[-2]
            self.estimator['regressor'].gamma = self.bestX[-1]
        
                    
        print('\r\n Fitting the MOSGO module with the best parameter combination...')
        
        # self.estimator.fit(self.X,self.y)
        if self.int_mode == 'opt':
            self.Support = self.bestX[:-3] == 1
        else:
            self.Support = np.round(self.bestX[:-3]).astype('int') == 1
        
        
        n_features = np.sum(self.Support)
        # lamb = 0.2
        bestScore = (n_features /self.nv)*self.lamb + 1 -self.bestf
            # nv = np.sum(x2[:-3])
        
            # f = (nv/self.nv)*lamb- meanscore
        # bestScore = 1-self.bestf 
        self.best_mosgo_result = {'hyperparameters': self.bestX[-3:],
                                  'n_features':n_features,
                                  'support':self.Support,
                                  'score':bestScore}
        
        # print('\n cv results: \n',self.rfe.cv_results)
        # print('\n rfe results: \n',self.rfe.rfe_results)
        print('\n mosgo results: \n',self.best_mosgo_result)
        
        Xtrain = X[:,self.Support]
        
        if self.yScale:
            if self.PreProcessor:
                y = self.PreProcessor.fit_transform(y)
        # self.estimator.fit(Xtrain,y) 
        
        if str(type(self.estimator))[-5:-2] == 'SVR':
            self.estimator.C = self.best_mosgo_result['hyperparameters'][-3]
            self.estimator.epsilon = self.best_mosgo_result['hyperparameters'][-2]
            self.estimator.gamma = self.best_mosgo_result['hyperparameters'][-1]
            self.estimator.fit(Xtrain,y)
        elif str(type(self.estimator))[-10:-2] == 'Pipeline':
            self.estimator['regressor'].C = self.best_mosgo_result['hyperparameters'][-3]
            self.estimator['regressor'].epsilon = self.best_mosgo_result['hyperparameters'][-2]
            self.estimator['regressor'].gamma = self.best_mosgo_result['hyperparameters'][-1]
            self.estimator.fit(Xtrain,y)
        
        # return self.best_mosgo_result
        

    def predict(self,X):
        
        if X.shape[1] != self.best_mosgo_result['n_features']:
            print('Since the shape of the input matrix is not equal to the best result,',
                  ' it regards the input as the best X and converts the size corresponding to the "support"')
            X = X[:,self.best_mosgo_result['support']]
        
        y_pred = self.estimator.predict(X)
        if self.yScale:
            if self.PostProcessor:
                y_pred = self.PostProcessor.fit_transform(y_pred)
        
        return y_pred
    
    def export_estimator(self,
               model_path='mosgo_svr.pkl'
               ):
        
        dump(self.estimator, model_path) 
        
        return
        
    def import_estimator(self,
               model_path='mosgo_svr.pkl'
               ):
        
        self.estimator = load(model_path) 
        
        return

    
    def get_Pareto_fronts(self,plot_opt=True):
        
        
        # Xarray = np.array(self.optimizer.Xs)
        # farray = np.array(self.optimizer.fs)
        
        varXarray = self.Xarray[:,:,:-3]
        nvXarray = np.sum(varXarray,axis=2).astype('int')
        Xunique = np.unique(nvXarray).astype('int')
        Xticks = np.arange(1,Xunique.max()+1,1)
        
        fPareto = np.array([self.farray[nvXarray==i].min() for i in Xunique])
        
        ParetoDiff = np.diff(np.append(fPareto.max()+1,fPareto))<0
        # Pareto_loc = np.where(np.diff(np.append(10**5,fPareto))<0)
        
        Pareto_loc = []
        i = 0
        while True:
            
            if ParetoDiff[i]:
                
                Pareto_loc.append(i)
                i += 1
            else:
                break
                    
        Xunique = Xunique[Pareto_loc]
        fPareto = fPareto[Pareto_loc]
        
        
        if plot_opt is True:
            from matplotlib import pyplot as plt
            
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.grid()
            ax.scatter(nvXarray.ravel(),self.farray,
                       edgecolor='k',facecolor='none',s=8)
            ax.plot(Xunique,fPareto,color='r',marker='o')
            
            # fmodified = fPareto + (Xunique/Xarray[:,:,:-3].shape[2])*0.5
            # fmodified = fPareto + (Xunique)*0.03
            # ax.plot(Xunique,fmodified,color='blue',marker='o')
            
            ax.set_ylim([0,1])
            ax.set_xticks(Xticks)
            ax.set_xlabel('The number of variables')
            # ax.set_ylabel('Fitness ($R^2$)')
            if self.lamb>0:
                ax.set_ylabel('Objective function (1-$R^2_{CV}+\lambda_v(n_v/n_{v,tot}) $)')
            else:
                ax.set_ylabel('Objective function (1-$R^2_{CV}$)')
            ax.set_title('Pareto front plot ($R^2_{CV}$>0)')
            ax.legend(['Population','Pareto front'])
        
        return Xunique, fPareto
    
    
    def fit_best(self,X,y,nv=None):    
        
        
        
        
        if nv is None:
            Xtrain = X[:,self.best_mosgo_result['support']]
            if self.yScale:
                if self.PreProcessor:
                    y = self.PreProcessor.fit_transform(y)
                
            if str(type(self.estimator))[-5:-2] == 'SVR':
                self.estimator.C = self.best_mosgo_result['hyperparameters'][0]
                self.estimator.epsilon = self.best_mosgo_result['hyperparameters'][1]
                self.estimator.gamma = self.best_mosgo_result['hyperparameters'][2]
                self.estimator.fit(Xtrain,y)
            elif str(type(self.estimator))[-10:-2] == 'Pipeline':
                self.estimator['regressor'].C = self.best_mosgo_result['hyperparameters'][0]
                self.estimator['regressor'].epsilon = self.best_mosgo_result['hyperparameters'][1]
                self.estimator['regressor'].gamma = self.best_mosgo_result['hyperparameters'][2]
                self.estimator.fit(Xtrain,y)
            
        else:
            Xunique,fPareto = self.get_Pareto_fronts(plot_opt=False)
            # Xarray = np.array(self.optimizer.Xs)
            # farray = np.array(self.optimizer.fs)
            nvXarray = np.sum(self.Xarray[:,:,:-3],axis=2).astype('int')
            best_mosgo_result = self.Xarray[(nvXarray == nv)&(self.farray == fPareto[Xunique==nv])][0]
            # if best_mosgo_result.ndim>1:
            #     best_mosgo_result = best_mosgo_result[0]
            # print(best_mosgo_result)
            support = best_mosgo_result[:-3].astype('bool')
            Xtrain = X[:,support]
            best_mosgo_result[self.X_log_flag] = 10**best_mosgo_result[self.X_log_flag]
            C = best_mosgo_result[-3]
            epsilon = best_mosgo_result[-2]
            gamma = best_mosgo_result[-1]
            print(Xtrain.shape)
            if self.yScale:
                if self.PreProcessor:
                    y = self.PreProcessor.fit_transform(y)
                
            if str(type(self.estimator))[-5:-2] == 'SVR':
                self.estimator.C = C
                self.estimator.epsilon = epsilon
                self.estimator.gamma = gamma
                self.estimator.fit(Xtrain,y)
            elif str(type(self.estimator))[-10:-2] == 'Pipeline':
                self.estimator['regressor'].C = C
                self.estimator['regressor'].epsilon = epsilon
                self.estimator['regressor'].gamma = gamma
                self.estimator.fit(Xtrain,y)
                
            print('Refitted the model with best parameter set for nv=%d \r\n Check the best MOSGO result'%(nv))
            print(best_mosgo_result)
            return best_mosgo_result
        

    def get_support_hyperparameters(self,best_mosgo_result=None):
        # X is the optimization solution
        if best_mosgo_result is not None:
            support = best_mosgo_result[:-3].astype('bool')
            hyperparameters = best_mosgo_result[-3:]
            
        else:
            support = self.best_mosgo_result['support']
            hyperparameters = self.best_mosgo_result['hyperparameters']
            
        return support, hyperparameters
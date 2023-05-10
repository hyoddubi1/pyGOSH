# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:49:04 2022

@author: user
"""

#Metaheuristic otimization algorithms
# Hyoseb Noh
# Update date : 2022 06 22
#%%
from .Metrics import neg_R2, neg_adjusted_R2, MSE, RMSE, MAE
import numpy as np
from pyDOE import lhs
import random
#from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
#from joblib import wrap_non_picklable_objects
#%%

def testprint():
    
    
    y = np.array([1,2,3,4,5,6,7,8,9,10])
    y_pred = np.array([1.2,1,4,4,5.2,5.8,7.3,7.5,9,10])
    
    return print(MAE(y,y_pred))

def MCCE(X,f,ub,lb,obj_func,alpha = 1.0, beta = 0.5,obj_eval='serial'):
    
    # alpha and beta are the 
    n, d = X.shape
    
    fcal = 0
    
    q = np.max([d, 2])
    q = np.min([n, q])
    
    sortidx = np.argsort(f)
    f = f[sortidx]
    X = X[sortidx]
    idxs = (np.arange(len(X))+1)
    rank = idxs[::-1]
    # Triangular probability
    p = 2 * rank / (n * (n+1)) 
    # Randomly choosing point
    
    for i in range(len(X)):
    
        r = np.random.choice(idxs,size=q,p = p)
        r = np.sort(r)-1
        ui = X[r]
        vi = f[r]
        
        #best point
        sb = ui[0]
        fb = vi[0]
        # worst point
        uq = ui[-1]
        vq = vi[-1]
        # N point second worst point
        uqN = ui[-2]
        vqN = vi[-2]
        
        ce = np.mean(ui[:-1],axis = 0)
        
            #reflection step
        Xnew = ce + alpha * (ce - uq)
        # print('Xnew before: \r\n',Xnew)
        Xnew = BoundaryHandling(Xnew,ub,lb)
        # print('\r\nXnew',Xnew)
        if obj_eval=='serial':
            fnew = obj_func(Xnew)
        elif obj_eval=='vector':
            fnew = obj_func(Xnew.reshape(1,-1))
        
        fcal = fcal + 1
        
        
        if fnew < vqN:
            if fnew < fb:
                Xnew1 = Xnew + alpha * (Xnew- ce)
                Xnew1 = BoundaryHandling(Xnew1,ub,lb)
                if obj_eval=='serial':
                    fnew1 = obj_func(Xnew1)
                elif obj_eval=='vector':
                    fnew1 = obj_func(Xnew1.reshape(1,-1))
#                fnew1 = obj_func(Xnew1.reshape(1,-1))
                fcal = fcal + 1
                if fnew1 < fnew:
                    fnew = fnew1.copy()
                    Xnew = Xnew1.copy()
        else:
            #contraction step
            if fnew > vq:
                Xnew = uq + beta * (ce - uq)
                Xnew = BoundaryHandling(Xnew,ub,lb)
                if obj_eval=='serial':
                    fnew = obj_func(Xnew)
                elif obj_eval=='vector':
                    fnew = obj_func(Xnew.reshape(1,-1))
#                fnew = obj_func(Xnew.reshape(1,-1))
                fcal = fcal + 1
                
                #mutation step
                if fnew>vq :
                    
#                    Xnew = lb + np.random.rand(d) *(ub-lb)
                    # print(X)
                    
                    if d == 1:
                        sig = np.sqrt(np.cov(np.squeeze(X)))
                        Xnew = np.random.normal(ce,sig,1)
                        # print('X',X,'ce',ce,'Xnew',Xnew)
                    else:
                        sig = np.cov(X, rowvar=False)    
                    # print(sig)
                        Dia = np.diag(sig)
                        sig = np.diag((Dia + np.mean(Dia))*2)
                    
                        Xnew = np.random.multivariate_normal(ce,sig,1)[0]
                    Xnew = BoundaryHandling(Xnew,ub,lb)
                    if obj_eval=='serial':
                        fnew = obj_func(Xnew)
                    elif obj_eval=='vector':
                        fnew = obj_func(Xnew.reshape(1,-1))
#                    fnew = obj_func(Xnew.reshape(1,-1))
                    fcal = fcal + 1
                    
            else:
                Xnew1 = ce + beta * (Xnew- ce)
                Xnew1 = BoundaryHandling(Xnew1,ub,lb)
                if obj_eval=='serial':
                    fnew1 = obj_func(Xnew1)
                elif obj_eval=='vector':
                    fnew1 = obj_func(Xnew1.reshape(1,-1))
#                fnew1 = obj_func(Xnew1.reshape(1,-1))
                fcal = fcal + 1
                if fnew1 < fnew:
                    fnew = fnew1.copy()
                    Xnew = Xnew1.copy()
    
        
        X[r[-1]] = Xnew
        f[r[-1]] = fnew
#        print('fnew',fnew)
    return X, f, fcal

def CCE(X,f,ub,lb,obj_func,alpha = 1.0, beta = 0.5,obj_eval='serial'):
    
    # alpha: reflection coefficient
    # beta: contraction coefficient
    n, d = X.shape
    
    fcal = 0
    
    q = np.max([d, 2])
    q = np.min([n, q])
    
    sortidx = np.argsort(f)
    f = f[sortidx]
    X = X[sortidx]
    idxs = (np.arange(len(X))+1)
    rank = idxs[::-1]
    # Triangular probability
    p = 2 * rank / (n * (n+1)) 
    # Randomly choosing point
    
    for i in range(len(X)):
    
        r = np.random.choice(idxs,size=q,p = p)
        r = np.sort(r)-1
        ui = X[r]
        vi = f[r]
        
        # worst point
        uq = ui[-1]
        vq = vi[-1]

        ce = np.mean(ui[:-1],axis = 0)
            #reflection step

        Xnew = ce + alpha * (ce - uq)

        Xnew = BoundaryHandling(Xnew,ub,lb)

        if obj_eval=='serial':
            fnew = obj_func(Xnew)
        elif obj_eval=='vector':
            fnew = obj_func(Xnew.reshape(1,-1))
        
        fcal = fcal + 1
           
        #contraction step
        if fnew > vq:
            Xnew = uq + beta * (ce - uq)
            Xnew = BoundaryHandling(Xnew,ub,lb)
            if obj_eval=='serial':
                fnew = obj_func(Xnew)
            elif obj_eval=='vector':
                fnew = obj_func(Xnew.reshape(1,-1))
            fcal = fcal + 1
            
            #mutation step
            if fnew>vq:
                
                Xnew = lb + np.random.rand(d) *(ub-lb)
                Xnew = BoundaryHandling(Xnew,ub,lb)
                if obj_eval=='serial':
                    fnew = obj_func(Xnew)
                elif obj_eval=='vector':
                    fnew = obj_func(Xnew.reshape(1,-1))
                fcal = fcal + 1
    
        
        X[r[-1]] = Xnew
        f[r[-1]] = fnew
    
    return X, f, fcal

def DE(X,f,ub,lb,obj_func,F = 1, CR = 0.75,mutation_scheme = 'best1',obj_eval='serial'):
    # Differential evolution
    # F: mutation rate
    # CR: crossover rate
    n, D = X.shape
#    newX = X.copy()
    uis = X.copy()
    newf = f.copy()
#    newf2 = f.copy()
    fcal = 0
    
    if mutation_scheme[:4] == 'best' or mutation_scheme == 'current_to_best':
        Xbest = X[np.argmin(f)]
    
    for i, Xi in enumerate(X):
        r = random.sample(range(n), 5)
        
        if mutation_scheme == 'best1':
            vi = Xbest + F*(X[r[0]] - X[r[1]])
        elif mutation_scheme == 'rand1':
            vi = X[r[2]] + F*(X[r[0]] - X[r[1]])
        elif mutation_scheme == 'rand2':
            vi = X[r[2]] + F*(X[r[0]] - X[r[1]]) + F*(X[r[3]] - X[r[4]])
        elif mutation_scheme == 'best2':
            vi = Xbest + F*(X[r[0]] - X[r[1]]) + F*(X[r[2]] - X[r[3]])
        elif mutation_scheme == 'current_to_best':
            vi = Xi + F*(Xbest - Xi) + F * (X[r[0]] - X[r[1]])
        elif mutation_scheme == 'current_to_rand':
            vi = Xi + F*(X[r[0]] - Xi) + F * (X[r[1]] - X[r[2]])
        
        rij = np.random.rand(D)    
#        for i in range(D):
        uis[i][rij<CR] = vi[rij<CR]
        uis[i] = BoundaryHandling(uis[i],ub,lb)
#        print(uis[i])
        
        if obj_eval=='serial':
            newf[i] = obj_func(uis[i])
        elif obj_eval=='vector':
            uii2d = uis[i].reshape(1,D)
            newf[i] = obj_func(uii2d)
        
        
        
        fcal = fcal + 1
            
    
    changed = f>newf
    X[changed] = uis[changed]
    f[changed] = newf[changed]
    
    
    return X, f, fcal

def DimensionRestore(obj_func,S,Sf,ub,lb,fcal,obj_eval='serial'):
    
    N, Dim = S.shape
    Snew = S.copy()
    Sfnew = Sf.copy()
    Nmean = np.mean(S, axis = 0)
    Nstd = np.std(S, axis = 0)
    a = S.transpose()

    for i in range(Dim):
        a[i,:] = (a[i,:] - Nmean[i])/(Nstd[i]+0.0001)
    
    r = np.max(np.max(a,axis = 0) - np.min(a,axis = 0))
    c = np.matmul(a,a.transpose()) / N
    
    d, v=np.linalg.eig(c)

    d = d / np.sum(d)
    lastdim = np.sum(d > (0.01/Dim))
    nlost = Dim - lastdim

    if nlost > np.floor(Dim/10)+1:
        for i in range(lastdim,Dim):
            happen = 0
            stemp = ((np.random.randn(1)+2)*r*v[:,i]).transpose()
            for j in range(Dim):
                stemp[j] = stemp[j]*Nstd[j] + Nmean[j]
            stemp = BoundaryHandling(stemp,ub,lb)
            if obj_eval=='serial':
                ftemp = obj_func(stemp)
            elif obj_eval=='vector':
                ftemp = obj_func(stemp.reshape(1,-1))
#            ftemp = obj_func(stemp.reshape(1,-1))
            fcal = fcal +1
            
#            print('stemp',stemp,'\n v:',v)
            
            if ftemp > np.max(Sfnew):
                stemp = ((np.random.randn(1)-2)*r*v[:,i]).transpose()
                for j in range(Dim):
                    stemp[j] = stemp[j]*Nstd[j] + Nmean[j]
                    
                stemp = BoundaryHandling(stemp,ub,lb)
                if obj_eval=='serial':
                    ftemp = obj_func(stemp)
                elif obj_eval=='vector':
                    ftemp = obj_func(stemp.reshape(1,-1))
#                ftemp = obj_func(stemp.reshape(1,-1))
                fcal = fcal +1
                
            if ftemp <np.max(Sfnew):
                happen = 1
                Snew[N-1,:] = stemp
                Sfnew[N-1] = ftemp
                
            if happen == 1:
                sortidx = np.argsort(Sfnew)
                Sfnew = Sfnew[sortidx]
                Snew = Snew[sortidx]
            
    return Snew, Sfnew, fcal

def BoundaryHandling(X,ub,lb):
    
    bound = ub - lb
    
    # upper bound reflection
    idx = X > ub
    rm = np.mod(X[idx]-ub[idx],bound[idx])
    X[idx] = ub[idx] - np.abs(rm)
    #lower bound reflection
    idx =  X < lb
    rm = np.mod(X[idx]-lb[idx],bound[idx])
    X[idx] = ub[idx] - np.abs(rm)
    
    return X
    
def Pop_init(n_pop,n_dim,lb=None,ub=None,method = 'uniform',seed = None):
    # Initialization of the population with min-max scaling
    np.random.seed(seed)
    if type(lb) != np.ndarray :
        lb = np.zeros(n_dim)
    else:
        lb = np.array(lb)
    if type(ub) != np.ndarray :
        ub = np.ones(n_dim)
    else:
        ub = np.array(ub)

    if method == 'uniform':
        pop = np.random.rand(n_pop,n_dim)
    elif method == 'Gaussian':
        pop = np.random.randn(n_pop,n_dim)
    elif method == 'LHS': # pyDOE Latin Hypercube sampling 
        pop = lhs(n_dim,n_pop)  

    if method == 'Gaussian':
        ubt = np.ceil(np.max(pop,axis = 0))
        lbt = np.floor(np.min(pop,axis = 0))  
    else:
        ubt = 1
        lbt = 0

    pop = lb + (pop-lbt) * (ub-lb) / (ubt- lbt)
    # if n_dim == 1:
    #     pop = np.array([pop])
    return pop
    

def dummy_func(x):
    return x**2

class Optimizer(object):
    def __init__(self,
            obj_func=dummy_func,
            lb = [0,0],
            ub = [1,1],
            algorithm = 'MCCE',
            stop_step = 50,
            stop_span = 10**-7,
            stop_obj_percent = 0.1,
            stop_fcal = 10000,
            dimension_restore = True,
            n_complex = 4,
            n_complex_size = None,
            iseed = None,
            iniflg = None,
            pop_init_method = 'LHS',
            init_pop = None,
            verbose = True,
            algorithm_params = 'default',
            n_jobs = 1,
            pre_dispatch = '2*n_jobs',
            obj_eval = 'serial',
            ):
        
        # If n_complex is larger than 1, this optimizer performs the 
        # SCE procedure automatically 
        
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.bound = np.array(ub)-np.array(lb)
        self.n_dim = len(self.lb)
        
        self.n_complex = n_complex
        if n_complex_size != None:
            self.n_complex_size = n_complex_size
        else:
            self.n_complex_size = 2 * self.n_dim + 1
        
        self.n_pop = self.n_complex * self.n_complex_size
        self.algorithm = algorithm

        self.pop_init_method = pop_init_method
        
        if type(init_pop) != np.ndarray :
            self.X = Pop_init(self.n_pop,self.n_dim,lb=self.lb,ub=self.ub,method = self.pop_init_method,seed = iseed)
        else:
            self.X = init_pop
            
        self.obj_func = obj_func
        self.verbose = verbose
        
        self.Xs = []
        self.fs = []
        self.bestXs = []
        self.bestfs = []
        
        self.stop_step = stop_step
        self.stop_span = stop_span
        self.stop_obj_percent =  stop_obj_percent
        self.stop_fcal = stop_fcal
        
        self.dimension_restore = dimension_restore
        
        self.fcal = 0
        
        if algorithm == 'DE':
            if algorithm_params == 'default':
                self.algorithm_params =  {'F':  1, 'CR' : 0.75, 'mutation_scheme' : 'best1'}
            else:
                self.algorithm_params = algorithm_params
        
        elif algorithm == 'CCE' or algorithm == 'MCCE':
            if algorithm_params == 'default':
                self.algorithm_params =  {'alpha':  1, 'beta' : 0.5}
            else:
                self.algorithm_params = algorithm_params
        
        
        
        
        self.obj_eval = obj_eval # Objective function evaluation procedure = 'serial' or 'vector'
        
        self.n_jobs = n_jobs
        if self.n_jobs > 1:
            self.pre_dispatch = pre_dispatch
            # self.parallel = Parallel(n_jobs=n_jobs,pre_dispatch = 4)
            self.parallel = Parallel(n_jobs=n_jobs)
#            self.parallel = Parallel(n_jobs=n_jobs,pre_dispatch = self.pre_dispatch)
#            print(self.parallel)
        if self.verbose:
            print('\n Initialized population\n')
            
            
    def _evolve(self,X,f):
        
        if self.dimension_restore:
            X,f,self.fcal = DimensionRestore(self.obj_func,X,f,self.ub,self.lb,self.fcal)
        
        if self.algorithm == 'CCE':
            newX, newf, fcal = CCE(X,f,self.ub,self.lb,
                             self.obj_func,
                             alpha = self.algorithm_params['alpha'],
                             beta = self.algorithm_params['beta'],
                             obj_eval=self.obj_eval)
            
        elif self.algorithm == 'MCCE':
            newX, newf, fcal = MCCE(X,f,self.ub,self.lb,
                             self.obj_func,
                             alpha = self.algorithm_params['alpha'],
                             beta = self.algorithm_params['beta'],
                             obj_eval=self.obj_eval)
            
        elif self.algorithm == 'DE':
            newX, newf, fcal = DE(X,f,self.ub,self.lb,
                            self.obj_func,
                            F = self.algorithm_params['F'],
                            CR = self.algorithm_params['CR'],
                            mutation_scheme = self.algorithm_params['mutation_scheme'],
                            obj_eval=self.obj_eval)
        
        self.fcal = self.fcal + fcal
        return newX, newf


    def evolve(self):
        
        if self.verbose:
            print('Evolve the population \n')
        
        X = self.X.copy()
        # print('X:\n',X)
        if self.obj_eval == 'serial':
            
            f = np.zeros((self.n_pop))
            for i in range(self.n_pop):
                # print(X[i])
                f[i] = self.obj_func(X[i])
        elif self.obj_eval == 'vector':
            f = self.obj_func(X)
        self.Xs.append(X)
        self.fs.append(f)
        
        best_pos = np.argmin(f)
        bestf = f[best_pos]
        bestX = X[best_pos]
        self.bestfs.append(bestf)
        self.bestXs.append(bestX)
        if self.n_complex >1:
            n_pop = self.n_pop
            d = self.n_dim
            
            CX = np.zeros([self.n_complex,self.n_complex_size,d])
            FX = np.zeros([self.n_complex,self.n_complex_size])
        
        count = 0
        while True:
            if self.verbose:
                print(str(count+1),'times evolution.\n')

            # Evaluate the objective function and Update the population
            if self.n_complex == 1:
                X, f = self._evolve(X,f) # dimension storing process is in this _evolve function
            else:
                #SCE procedure

                sortidx = np.squeeze(np.argsort(f,axis = 0))

                f = f[sortidx]
                X = X[sortidx]

                for ic in range(self.n_complex_size):
                    K1 = np.array(random.sample(range(self.n_complex), self.n_complex)) + ic * self.n_complex
                    K1 = np.squeeze(K1)

                    CX[:,ic,:] = X[K1,:]

                    FX[:,ic] = np.squeeze(f[K1])
                    
                if self.n_jobs > 1:
                    # with parallel_backend("loky"):

                        # CX[ic], FX[ic] = self.parallel(
                        #     delayed(self._evolve)(
                        #         CX[ic],FX[ic]
                        #     )
                        #     for ic in range(self.n_complex)
                        # )
                        # Results = Parallel(n_jobs = self.n_jobs)(
                        #     delayed(self._evolve)(
                        #         CX[ic],FX[ic]
                        #     )
                        #     for ic in range(self.n_complex)
                        # )

                        # for ic in range(self.n_complex):
                        #     CX[ic] = Results[ic][0]
                        #     FX[ic] = Results[ic][1]
                    with parallel_backend("threading"):
                        Results = Parallel(n_jobs = self.n_jobs)(
                            delayed(self._evolve)(
                                CX[ic],FX[ic]
                            )
                            for ic in range(self.n_complex)
                        )

                        for ic in range(self.n_complex):
                            CX[ic] = Results[ic][0]
                            FX[ic] = Results[ic][1]
                        
                else:
               
                    for ic in range(self.n_complex):
                        CX[ic], FX[ic] = self._evolve(CX[ic],FX[ic])
                    
                X = CX.reshape(n_pop,d)
                f = FX.reshape(n_pop,)
                # print('Xafter:',X.shape)
            
            best_pos = np.argmin(f)
            bestf = f[best_pos]
            bestX = X[best_pos]
            if self.verbose:
                print('Best X: ',bestX,', Best f: ',bestf, '\n')
            self.Xs.append(X.copy())
            self.fs.append(f.copy())
            self.bestfs.append(bestf)
            self.bestXs.append(bestX)
            
            if self.fcal > self.stop_fcal:
                if self.verbose:
                    print('Evolution stopped since the X span is less than the stop criterion.\n')
                break

            span = np.exp(np.mean(np.log1p((np.max(X,axis = 0)-np.min(X,axis = 0))/np.mean(self.ub-self.lb)))) -1
            if span < self.stop_span:
                if self.verbose:
                    print('Evolution stopped since the X span is less than the stop criterion.\n')
                break
            if count > self.stop_step:
                obj_improve = np.abs(self.bestfs[-1] - self.bestfs[-2])          
                obj_improve = 100*obj_improve / np.mean(np.abs(self.bestfs[-self.stop_step:]))
                if obj_improve < self.stop_obj_percent:
                    if self.verbose:
                        print('Evolution stopped since objective function improvement is less than the stop criterion.\n')
                    break
            
            count = count+1
            if count == self.stop_step:
                if self.verbose:
                    print('Evolution stopped since the maximum evolution number was achieved.\n')
                break
        
        return bestX, bestf
    
    def FreezeAndGo(self):
        #Check whether the evolution stop is due to the stop_step.
        self.X = self.Xs[-1]
        bestX, bestf = self.evolve()
        return bestX, bestf
        
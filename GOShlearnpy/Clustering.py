# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:19:20 2021

@author: user
"""

import numpy as np

import random

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon   
import matplotlib.patheffects as pe

import matplotlib.cm as cm
import matplotlib.colors as mapcolors
#%%  Subroutines

def l2norm(v):
    # vector size calculator by l2-norm
    v = np.array(v)
    
    
    
    if np.ndim(v) == 1:
        v_size = 0
        for i in range(np.shape(v)[0]):
    
            v_size = v_size + v[i]**2
            
    else:
        v_size = np.zeros(len(v))
        for i in range(np.shape(v)[1]):
    
            v_size = v_size + v[:,i]**2

    v_size = np.sqrt(v_size)
            
    return v_size
#
#def Norm(v,Order = 2):
#    # vector size calculator by l-Order-norm
#    # Order: int, default = 2 : The order of the norm
#    
#    v = np.array(v)
#    
#    v_size = np.zeros(len(v))
#    
#    for i in range(np.shape(v)[1]):
#
#        v_size = v_size + v[:,i]**Order
#
#    v_size = v_size ** (1/Order)
#            
#    return v_size


def multivariate_Gaussian(x,mu,Sigma):
#     if x be a column vector (N_vars,1), it returns one probability
#     else if  x be a (N_vars,N_Data) matrix, it return probabilities of each data point
    
    D = len(mu)
    
    Sigma_inv = np.linalg.inv(Sigma)
    
    pcoef = 1 / (2*np.pi)**(D/2) / np.linalg.det(Sigma)**(0.5)
    
   
    x_mu = x - mu

    p =pcoef * np.exp(-1/2 *(np.diag(x_mu.T @ Sigma_inv @ x_mu)))
    
    
    return p


def MinMaxScaler(Dataset):
    Scaled = (Dataset - np.min(Dataset,0) )/(np.max(Dataset,0)  - np.min(Dataset,0))
    
    return Scaled

def MinMaxScaler_rev(Dataset,Scaled):
    Scaled2 = Scaled *(np.max(Dataset,0)  - np.min(Dataset,0)) +  np.min(Dataset,0)
    
    return Scaled2


def DBI(Dataset,n_clusters, ClusterLabel,Centroids):
# DaviesBouldinIndex
    SCs = np.zeros(n_clusters)
    Rij = np.zeros([n_clusters,n_clusters])
    dces = np.zeros([n_clusters,n_clusters])
    
    
    for i in range(0,n_clusters):
        Qi = Dataset[ClusterLabel==i]
        SCs[i] = np.mean(l2norm(Qi - Centroids[i]))
    
        for j in range(0,n_clusters):
            
            dces[i][j] = l2norm(Centroids[i]-Centroids[j])
            
            
    for i in range(0,n_clusters):
        
        for j in  range(0,n_clusters):
            
            if i != j:
                                              
                Rij[i][j] = (SCs[i] + SCs[j]) / dces[i][j]
                
    Ri = [np.max(Rij[i]) for i in range(0,n_clusters)]
            
            
            
    DBI = np.mean(Ri)
        
        
    return DBI

def QE(Dataset,Weight):
    
    NCol = np.shape(Weight)[1]
    
    n_data = np.size(Dataset,0)
    
    QE = 0
    for temp_Data in  Dataset:
    #    print(temp_data)
        
        Dists = np.linalg.norm((Weight - temp_Data),axis=2)
        
        argmindist = np.argmin(Dists)
        
        WinningNode = [int(argmindist / NCol),argmindist % NCol]
        
        
        QE = QE + np.linalg.norm(Weight[WinningNode[0],WinningNode[1]] - temp_Data)
        QE = QE / n_data
        
    return QE

def TE(Dataset,Weight):
    NCol = np.shape(Weight)[1]
    
    n_data = np.size(Dataset,0)

    TE = 0    
    
    for temp_Data in  Dataset:
    #    print(temp_data)
        
        Dists = np.linalg.norm((Weight - temp_Data),axis=2)
        
        argmindist = np.argmin(Dists)
        
        WinningNode = [int(argmindist / NCol),argmindist % NCol]
        
        Dist_2ndmin = np.min(Dists[Dists>Dists[WinningNode[0],WinningNode[1]]])
        
        where_2nd = np.where(Dists == Dist_2ndmin)
        
        WinningNode_2nd = [where_2nd[0][0],where_2nd[1][0]]

        
        if np.max(np.abs(np.array(WinningNode_2nd) -np.array(WinningNode)))>1:
            TE = TE + 1
    TE = TE / n_data
    
    return TE

def rgb_to_hex(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

#%% k-means clustering class
    
class Kmeans(object):
    def __init__(self,n_clusters = 3, Centroid_method = 'mean',tol = 10**(-5), max_iter = 100):
        self.tol = tol
        self.n_clusters = n_clusters
        self.Centroid_method = Centroid_method
        self.max_iter = max_iter
        
        
    def train(self,Dataset):
        iters = 0
        self.n_data = np.shape(Dataset)[0]
        self.ClusterLabel = np.zeros(self.n_data)
                
        preClusterLabel = self.ClusterLabel + 1
        res = 1 # dummy residual        
        
        self.Centroids = Dataset[random.sample(range(0,len(Dataset)),self.n_clusters)]
        
        dis = np.zeros([self.n_clusters,self.n_data])
        
        while self.tol<res:
            iters = iters + 1
            for j in range(self.n_clusters):
                dis[j] = l2norm(Dataset - self.Centroids[j])
                
            self.ClusterLabel = np.argmin(dis,axis = 0)
            
            for i in range(0,self.n_clusters):  
                if self.Centroid_method == 'mean':
                    self.Centroids[i] = np.mean(Dataset[self.ClusterLabel==i],axis=0)
                    
                elif self.Centroid_method == 'median':
                    self.Centroids[i] = np.median(Dataset[self.ClusterLabel==i],axis=0)
        
            res =  np.max(preClusterLabel-self.ClusterLabel)
            preClusterLabel = self.ClusterLabel
            if iters>self.max_iter:
                print("Evolution breaks: iters>max_iter")
                break
            
    def predict(self,Dataset):
        
        dis = np.zeros([self.n_clusters,self.n_data])
        for j in range(self.n_clusters):
                dis[j] = l2norm(Dataset - self.Centroids[j])
                
        self.ClusterLabel = np.argmin(dis,axis = 0)
        

            
#%% Gaussian mixture model class

class GMM(object):
    def __init__(self, n_clusters = 3, tol = 10**-3,Centroid_method = 'mean',
                 Init_method = 'kmeans', max_iter = 100):
        self.res = 10**10 
        self.tol = tol
        self.n_clusters = n_clusters
        self.Centroid_method = Centroid_method
        self.Init_method = Init_method
        self.max_iter = max_iter

        
    def Point_initialization(self,Dataset):
        
        [self.n_data, self.n_var] = np.shape(Dataset)
    
            
        if self.Init_method == 'random':
            self.mu = np.random.rand(self.n_clusters,self.n_var,1) # mean initialization
            self.Sigma = np.random.rand(self.n_clusters,self.n_var,self.n_var) +  np.eye(self.n_var)
            
        elif self.Init_method == 'kmeans':
            Kmeansclass = Kmeans(n_clusters = self.n_clusters,Centroid_method = self.Centroid_method)
            Kmeansclass.train(Dataset)            
            
            self.mu = np.reshape(Kmeansclass.Centroids,[self.n_clusters,self.n_var,1])
            self.Sigma = np.zeros([self.n_clusters,self.n_var,self.n_var])
            for i in range(self.n_clusters):
                self.Sigma[i] = np.cov(Dataset[Kmeansclass.ClusterLabel == i],rowvar=False)
            
        else:
            print('error:No such method for initialization-> random initialize')
            self.mu = np.random.rand(self.n_clusters,self.n_var,1) # mean initialization
            self.Sigma = np.random.rand(self.n_clusters,self.n_var,self.n_var) +  np.eye(self.n_var)
            
        self.pi_coef = np.ones([self.n_clusters,1])
        self.Prob = np.zeros([self.n_clusters,self.n_data])
        self.piProb = np.zeros([self.n_clusters,self.n_data])
        self.gamma = np.zeros([self.n_clusters,self.n_data])
        
        
        
    def EMalgorithm(self,Dataset):
        LL_pre = -10**6
        x = Dataset.T
        iters = 0
            # first probability calculation
        for k in range(self.n_clusters):
            self.Prob[k] = multivariate_Gaussian(x,self.mu[k],self.Sigma[k])
            self.piProb[k] = self.pi_coef[k] * self.Prob[k]
        
        while (self.tol<self.res):
            iters = iters + 1
#            print(iters)
        # E step    
        # evalutation of responsibilities
            for k in range(self.n_clusters):
                self.gamma[k] = self.piProb[k] / (np.sum(self.piProb,0))
    
            # M step
               
                Nk = np.sum(self.gamma[k])
    
                self.mu[k] = 1 / Nk *np.atleast_2d(np.sum(self.gamma[k]*x,1)).T
    
                Sigma_k = np.zeros([self.n_var,self.n_var])
                
                x_muk = x - self.mu[k]

                for i in range( self.n_var):
                    for j in range(i, self.n_var):
                        Sigma_k[i,j] = np.dot(self.gamma[k] * x_muk[i],x_muk[j])
                        Sigma_k[j, i] = Sigma_k[i,j]
                    
    
                self.Sigma[k] = Sigma_k / Nk
    
                self.pi_coef[k] = Nk/self.n_data
    
            # Evaluation of log likelihood
#            for k in range(self.n_clusters):
                self.Prob[k] = multivariate_Gaussian(x,self.mu[k],self.Sigma[k])
                self.piProb[k] = self.pi_coef[k] * self.Prob[k]
    
            self.LL = np.sum(np.log(np.sum(self.piProb,0)))
                    
            self.res = np.abs(self.LL - LL_pre)
    
            LL_pre = self.LL
            
            if iters>self.max_iter:
                print("Evolution breaks: iters>max_iter")
                break
#         print(res)
        
        
        self.ClusterLabel = np.argmin(self.piProb,0)
        
        
        
    def train(self,Dataset):
        
        self.Point_initialization(Dataset)
        self.EMalgorithm(Dataset)  
        
    def predict(self,Dataset):
        [self.n_data, self.n_var] = np.shape(Dataset)
        self.Prob = np.zeros([self.n_clusters,self.n_data])
        self.piProb = np.zeros([self.n_clusters,self.n_data])
        
        
        for k in range(self.n_clusters):
            self.Prob[k] = multivariate_Gaussian(Dataset.T,self.mu[k],self.Sigma[k])
            self.piProb[k] = self.pi_coef[k] * self.Prob[k]
            self.ClusterLabel = np.argmin(self.Prob,0)
            
            
      
#%% Self-organizing map (SOM)

class SOM(object):
    def __init__(self,SOM_Dimension = [10,10],N_iter = 10000, Init_eta = 0.15, Init_sigma = 'auto'  ,epoch = 4, 
                 ToroidOpt = True, Init_method = 'pca',PrintOpt = True, TrainOpt = "batch", DecayOpt = "linear",
                 GridOpt = "rectangular", VarNames = []):
        
        self.SOM_Dimension = SOM_Dimension
        self.N_iter = N_iter 
        self.Init_eta = Init_eta 
        self.PrintOpt = PrintOpt
        
        if Init_sigma == 'auto':
            self.Init_sigma = max(SOM_Dimension)/2
        else:
            self.Init_sigma = Init_sigma
        
        self.epoch = epoch 
        
        self.ToroidOpt = ToroidOpt
        self.TrainOpt = TrainOpt
        self.DecayOpt = DecayOpt
        self.GridOpt = GridOpt
        self.Init_method = Init_method
        self.NRow = SOM_Dimension[0]
        self.NCol = SOM_Dimension[1]
        self.ClusteringClass = []
        self.WeightLabel=[]
        self.ClusterLabel_Weight=[]
        self.VarNames= VarNames
        
   
    def Weight_init(self,Dataset):     
        
        if self.Init_method == 'pca':

            pca = PCA(n_components=2)
            pca.fit(Dataset)
            pca_vectors_ = np.array([ pca.components_[i] * pca.explained_variance_[i] for i in range(len(pca.components_))])
            
            self.Weight = np.zeros((self.SOM_Dimension[0],self.SOM_Dimension[1],self.n_var))
            
            first_span = np.linspace(-pca_vectors_[0],pca_vectors_[0],np.max(self.SOM_Dimension))
    
            second_span = np.linspace(-pca_vectors_[1],pca_vectors_[1],np.min(self.SOM_Dimension))
    
            for i in range(self.SOM_Dimension[np.argmax(self.SOM_Dimension)]):
                for j in range(self.SOM_Dimension[np.argmin(self.SOM_Dimension)]):
                    if np.argmax(self.SOM_Dimension) == 1:
                        iw = j
                        jw = i
                    else:
                        iw = i
                        jw = j                
    
                    self.Weight[iw][jw] =  (first_span[i] + second_span[j])
            self.Weight = np.mean(Dataset,axis = 0) + self.Weight
    
        elif self.Init_method== 'random':
            self.Weight = np.random.rand(self.SOM_Dimension[0],self.SOM_Dimension[1],self.n_var) *  (np.max(Dataset,axis=0) - np.min(Dataset,axis=0)) +np.min(Dataset,axis=0)
            
    def Decay_Func(self,Init_Value,t,t_max,DecayOpt):
        
        if DecayOpt=="exponential":
            DecayedValue = Init_Value * np.exp(-2 * Init_Value * t / t_max)
        elif DecayOpt == "linear":
            DecayedValue = Init_Value * ((t_max - t )/ t_max)
        
        return DecayedValue
    
    def Update_Weight(self,Weight,C_Data,WinningNode,sigma,eta):
    
    #     [sigma eta] values are current effective radius and learning rate rspectively
        
        NRow = np.size(Weight,0)
        NCol = np.size(Weight,1)
        R1 = np.array(WinningNode)
        dW = Weight*0
        Lambda = dW
        
        R1[0]-sigma
        R1[0]+sigma
        
        if self.GridOpt == "hexagonal":
            if np.mod(R1[0],2) == 0:
                R1[1] = R1[1]  + 0.5
        
        for i in range(0,NRow):
    
            for j in range(0,NCol):
                
                if self.ToroidOpt == True:
    
                    R2candid1 = np.array([j,j+NCol,j-NCol])
                    R2subt1 = abs(R2candid1 - R1[1]  )
                    R21 = R2candid1[np.argmin(R2subt1)]
    
                    R2candid0 = np.array([i,i+NRow,i-NRow])
                    R2subt0 = abs(R2candid0 - R1[0]  )
                    R20 = R2candid0[np.argmin(R2subt0)]
                    
                    if self.GridOpt == "hexagonal":
                        if np.mod(R20,2) == 0:
                            R21 = R21 + 0.5
                    
                    R2 = [R20,R21]
                    
                else:
                    
                    R2 = [i,j]
                R2 = np.array(R2)
                tDist = l2norm(R1-R2)                
    
                if tDist < sigma:
    
                    Lambda[i][j] = np.exp(- l2norm(R1-R2) **2/ (2 * sigma**2))
    
        dW = eta * Lambda * (C_Data - Weight)
    
        Updated_Weight = Weight + dW  
    
    
        return Updated_Weight



    def Winning_node(self,Weight,C_Data,NCol):
        
        Dists = np.linalg.norm((Weight -C_Data),axis=2)

        argmindist = np.argmin(Dists)

        WinningNode = [int(argmindist / NCol),argmindist % NCol]
        
        return WinningNode
        

        
    def train_random(self,Dataset):
        for j in range(0,self.epoch):
            
            if self.PrintOpt:

                print("epoch: ",j+1)
                
            for i in range(0,self.N_iter):
    
        #         Decayed learning rate and neighborhood radius
                eta = self.Decay_Func(self.Init_eta,i,self.N_iter,self.DecayOpt)
                sigma = self.Decay_Func(self.Init_sigma,i,self.N_iter,self.DecayOpt)
    
        #         Randomly selected data point
                C_Data = Dataset[random.randrange(self.n_data)]        
    
        #         Finding winning node
    
                WinningNode = self.Winning_node(self.Weight,C_Data,self.NCol)
    
        #         Update weight network 
                self.Weight = self.Update_Weight(self.Weight,C_Data,WinningNode,sigma,eta)
        
        
        
    def train_batch(self,Dataset):
#        
        for j in range(0,self.epoch):
            if self.PrintOpt:

                print("epoch: ",j+1)         

            eta = self.Decay_Func(self.Init_eta,j,self.epoch,self.DecayOpt)
            sigma = self.Decay_Func(self.Init_sigma,j,self.epoch,self.DecayOpt)
#            print(sigma)
            
            BMUs = np.zeros((self.n_data,2))
            for nn in range(self.n_data):
#                reshaped_BMUs = np.argmin(l2norm(Dataset[nn] - re_weight),axis = 0)
#                BMUs[nn] = np.divmod(reshaped_BMUs,self.NCol)
                BMUs[nn] = self.Winning_node(self.Weight,Dataset[nn],self.NCol)
                
            SOM_means = self.Weight * 0    


            nj = np.zeros(self.NRow* self.NCol)
            rj = np.zeros((self.NRow* self.NCol,2))
            
            for ii in range(self.NRow):
                for jj in range(self.NCol):
                    Bool_Subset = (BMUs[:,0]==[ii])&(BMUs[:,1]==[jj])
                    Subset = Dataset[Bool_Subset]
                    if np.sum(Bool_Subset)  == 0:
                        SOM_means[ii][jj] = self.Weight[ii][jj]
                    elif np.sum(Bool_Subset)  == 1:
                        SOM_means[ii][jj] = Subset
                    else:
                        SOM_means[ii][jj] = np.mean(Subset,axis = 0)
            #        nj[ii,jj] = np.shape(Subset)[0]
                    nj[self.NCol*ii+jj] = np.shape(Subset)[0]
                    rj[self.NCol*ii+jj] = [ii,jj]
            
            nj[nj == 0] = 1 # To avoid singular point when the c / 0 due to nj == 0
            
            re_SOM_means = np.reshape(SOM_means,(self.NRow* self.NCol,self.n_var))
#            if j == 199 > 0:
#                print(rj)
            
            for ii in range(self.NRow):
                for jj in range(self.NCol):
                    
                    if self.ToroidOpt == True:
                        dist_temp = rj * 0
                        R2candid1 = np.array([rj[:,1],rj[:,1]+self.NCol,rj[:,1]-self.NCol])
                        
                        if self.GridOpt == "hexagonal":
                            if np.mod(ii,2) == 0:
                                R2subt1 = abs(R2candid1 - (jj+0.5) )
                            else:
                                R2subt1 = abs(R2candid1 - jj  )   
                                
                            
                        else:
                            R2subt1 = abs(R2candid1 - jj  )
                        dist_temp[:,1] = R2candid1[np.argmin(R2subt1,axis = 0),np.arange(len(rj))]
        
                        R2candid0 =np.array([rj[:,0],rj[:,0]+self.NRow,rj[:,0]-self.NRow])
                        R2subt0 = abs(R2candid0 - ii  )
                        dist_temp[:,0] = R2candid0[np.argmin(R2subt0,axis = 0),np.arange(len(rj))]
                        if self.GridOpt == "hexagonal":
                            dist_temp[:,0] = dist_temp[:,0] * np.sqrt(3)/2
                        
                        
                        hj = eta* np.exp(-l2norm(dist_temp - [ii,jj] )  **2/ (2 * sigma**2))
                        
#                        if j == 199 > 0:
#                            print(hj)
#                        
                    else:
                    
                        hj = eta* np.exp(-l2norm([ii,jj] - rj)  **2/ (2 * sigma**2))
                    
#                    hj[hj>sigma] = 0

                    self.Weight[ii,jj] = re_SOM_means.T @(hj * nj)/ np.dot(hj, nj)
                
#            if res < tol:
##            print(self.Weight)
        
        
    def train(self,Dataset):

        self.n_data = np.size(Dataset,0)
        self.n_var  = np.size(Dataset,1)    
        
        self.Weight_init(Dataset)
        
        if self.TrainOpt == "batch":
            self.train_batch(Dataset)
        elif self.TrainOpt == "random":
            self.train_random(Dataset)
          
        
        self.predict(Dataset)
            
            
    
    def predict(self,Dataset):
        
        self.WeightLabel = np.zeros((self.n_data,2))
        self.n_data = np.size(Dataset,0)
               
        for i in range(self.n_data):
            
            self.WeightLabel[i] = self.Winning_node(self.Weight,Dataset[i],self.NCol)
            
        self.WeightLabel = self.WeightLabel.astype(int)
        
        if np.size(self.ClusterLabel_Weight) != 0:
            self.ClusterLabel_Data = np.zeros(self.n_data).astype(int)
            for i in range(self.n_data):
                self.ClusterLabel_Data[i] = self.ClusterLabel_Weight[self.WeightLabel[i][0],self.WeightLabel[i][1]]
            
    def Clustering(self,n_clusters = 3,Method = "kmeans",max_iter = 5000,Init_method = 'kmeans'):
        
        re_weight = np.reshape(self.Weight,(self.NRow* self.NCol,self.n_var)) 
        
        if Method == "kmeans":
            self.ClusteringClass = Kmeans(n_clusters = n_clusters, max_iter = max_iter )
            self.ClusteringClass.train(re_weight)
#            re_ClusterLabel_Weight = self.Kmeansclass.ClusterLabel
            self.ClusterLabel_Weight = np.reshape(self.ClusteringClass.ClusterLabel,
                                                  (self.SOM_Dimension[0],self.SOM_Dimension[1]))
            
        elif Method == "GMM":
            self.ClusteringClass = GMM(n_clusters = n_clusters ,max_iter = max_iter,Init_method = Init_method)
            self.ClusteringClass.train(re_weight)
#            re_ClusterLabel_Weight = self.GMMclass.ClusterLabel
            self.ClusterLabel_Weight = np.reshape(self.ClusteringClass.ClusterLabel,
                                                  (self.SOM_Dimension[0],self.SOM_Dimension[1]))
            

        
    def plot_grid(self,SOM_Dimension = [10,10],WeightLabel=[],ClusterLabel_Weight=[], GridOpt = "rectangular"):
        
        if np.size(WeightLabel) == 0:
            if np.size(self.WeightLabel) != 0:
                WeightLabel= self.WeightLabel
            else:
                WeightLabel = np.zeros((self.n_data,2))
        WeightLabel = WeightLabel.astype(int)
        
        if np.size(ClusterLabel_Weight) == 0:
            if np.size(self.ClusterLabel_Weight ) != 0:
                ClusterLabel_Weight = self.ClusterLabel_Weight        
            else:
                ClusterLabel_Weight = np.zeros((self.SOM_Dimension[0],self.SOM_Dimension[1]))
                
        ClusterLabel_Weight = ClusterLabel_Weight.astype(int)
#        print(ClusterLabel_Weight)
        color_pallete = [['#1f77b4'], ['#ff7f0e'],[ '#2ca02c'], ['#d62728'], ['#9467bd'], ['#8c564b'], ['#e377c2'], ['#7f7f7f'], ['#bcbd22'],[ '#17becf']]

        n_points = []
        colors = []
        hcoord = []
        vcoord = []
        CLabel = []
        for ii in range(self.SOM_Dimension[0]):
            for jj in range(self.SOM_Dimension[1]):
#                print(ii)
                if GridOpt == "hexagonal":
                    if np.mod(ii,2) == 0:
                        hcoord.append(jj+0.5)
                    else:
                        hcoord.append(jj)
                    vcoord.append(ii*np.sqrt(3)/2)
                else:
                    hcoord.append(jj)
                    vcoord.append(ii)
                
                if np.size(WeightLabel) == 1: # if it is False, the size is 1
                    n_points.append(" ")
                else:
                    n_points.append(np.sum((WeightLabel[:,0] == ii) & (WeightLabel[:,1] == jj)))
                    
                if np.size(ClusterLabel_Weight) == 1: # if it is False, the size is 1
                    colors.append("Grey")
                    CLabel.append(" ")
                else:
                    colors.append(color_pallete[ClusterLabel_Weight[ii,jj]])
                    CLabel.append(ClusterLabel_Weight[ii,jj])
        
#        plt.style.use('presentation')
        fig, ax = plt.subplots(1,figsize=(self.SOM_Dimension[0], self.SOM_Dimension[1]))
        
        ax.set_aspect('equal')
        
        # Add some coloured hexagons
        for x, y, c, l,cl in zip(hcoord, vcoord, colors, n_points, CLabel):
            color = c[0].lower()  # matplotlib understands lower case words for colours
            if GridOpt == "hexagonal":
                hex = RegularPolygon((x, y), numVertices=6, radius=0.577, 
                                     orientation=np.radians(0), 
                                     facecolor=color, alpha =0.3, edgecolor='k')
            elif GridOpt == "rectangular":        
                hex = RegularPolygon((x, y), numVertices=4, radius=0.707, 
                                     orientation=np.radians(45), 
                                     facecolor=color, alpha =0.3, edgecolor='k')
            ax.add_patch(hex)
            # Also add a text label
            ax.text(x, y+0.3, l, ha='center', va='center', size=10)
            
            if np.max(ClusterLabel_Weight) !=0:
                ax.text(x, y, cl+1, ha='center', va='center', size=12,weight='bold',c = c[0].lower(),
                        path_effects=[pe.withStroke(linewidth=2, foreground='w')])
        
        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, c=[c[0].lower() for c in colors], alpha=0)
        ax.axis('off')

        plt.show()
                
        return fig, ax                
            
    def plot_heatmap(self, Weight = [],ClusterLabel_Weight=[],GridOpt = "rectangular",colormap = "hot"):
        

        titles = []
        
        for i in range(self.n_var):
                titles.append("CP " + str(i+1))
        if self.VarNames != []:
            for i in range(len(self.VarNames)):
                titles[i] = self.VarNames[i]
        
        if np.size(Weight) == 0:
            Weight = self.Weight
        if np.size(ClusterLabel_Weight) == 0:
            if np.size(self.ClusterLabel_Weight ) != 0:
                ClusterLabel_Weight = self.ClusterLabel_Weight        
            else:
                ClusterLabel_Weight = np.zeros((self.SOM_Dimension[0],self.SOM_Dimension[1])).astype(int)
        
            
        color_pallete = [['#1f77b4'], ['#ff7f0e'],[ '#2ca02c'], ['#d62728'], ['#9467bd'], ['#8c564b'], ['#e377c2'], ['#7f7f7f'], ['#bcbd22'],[ '#17becf']]

        
        hcoord = []
        vcoord = []
        colors_w = []
        CLabel = []
        for ii in range(self.SOM_Dimension[0]):
            for jj in range(self.SOM_Dimension[1]):
                if GridOpt == "hexagonal":
                    if np.mod(ii,2) == 0:
                        hcoord.append(jj+0.5)
                    else:
                        hcoord.append(jj)
                    vcoord.append(ii*np.sqrt(3)/2)
                else:
                    hcoord.append(jj)
                    vcoord.append(ii)
                    
                if np.size(ClusterLabel_Weight) == 1: # if it is False, the size is 1
                    colors_w.append("Grey")
                    CLabel.append(" ")
                else:
                    colors_w.append(color_pallete[ClusterLabel_Weight[ii,jj]])
                    CLabel.append(ClusterLabel_Weight[ii,jj])
                    
                    
        figs = []   
        axes = []                 

        for i in range(self.n_var):
            temp_var = Weight[:,:,i]
            
            
            fig, ax = plt.subplots(1,figsize=(self.SOM_Dimension[0], self.SOM_Dimension[1]))

            ax.set_aspect('equal')
            ax.set_xticks(np.arange(0,np.shape(ClusterLabel_Weight)[1]+1))
            ax.set_yticks(np.arange(0,np.shape(ClusterLabel_Weight)[0]+1))
            ax.axis('off')
            ax.set_title(titles[i],size=20)

            cmap = cm.ScalarMappable(None,colormap)
            norm = mapcolors.Normalize(vmin=temp_var.min(),vmax=temp_var.max())
            
            colors = []
            for ii in range(self.SOM_Dimension[0]):
                for jj in range(self.SOM_Dimension[1]):
                    
                    colors.append(cmap.cmap(norm(temp_var[ii,jj])))
            
            # Add some coloured hexagons
            for x, y, c, c_w, cl in zip(hcoord, vcoord, colors,colors_w,CLabel):
                
                if GridOpt == "hexagonal":
                    hex = RegularPolygon((x, y), numVertices=6, radius=0.577, 
                                         orientation=np.radians(0), 
                                         facecolor=c, edgecolor='k')
                elif GridOpt == "rectangular":        
                    hex = RegularPolygon((x, y), numVertices=4, radius=0.707, 
                                         orientation=np.radians(45), 
                                         facecolor=c, edgecolor = 'k')
                    
#                if (x-1)>=0:
                    
                    
                if np.max(ClusterLabel_Weight) !=0:
                    ax.text(x, y, cl+1, ha='center', va='center', size= 12,c = 'w',
                        path_effects=[pe.withStroke(linewidth=5, foreground=c_w[0].lower())])
                ax.add_patch(hex)
            
            sm = plt.cm.ScalarMappable(cmap=colormap,norm=norm)
            sm._A = []
#            plt.colorbar(sm,ticks=range(int(temp_var.min()),int(temp_var.max())+1))   
            plt.colorbar(sm,
                         ticks=np.arange(temp_var.min(),temp_var.max(),(temp_var.max() - temp_var.min())/10).round(1),
                         fraction=0.046, pad=0.04)   
#            plt.clim(-0.5, 5.5)
             
            ax.scatter(hcoord, vcoord, c=[c[0].lower() for c in colors_w],edgecolors = 'k') 
            figs.append(fig)
            axes.append(ax)

        plt.show()
        return figs, axes
            
            

    def plot_heatmap_one(self, Weight = [],ClusterLabel_Weight=[],GridOpt = "rectangular",
                         colormap = "hot",ncol_subplot = 4,PlotSize = 5):
        

        titles = []
        
        for i in range(self.n_var):
                titles.append("CP " + str(i+1))
        if self.VarNames != []:
            for i in range(len(self.VarNames)):
                titles[i] = self.VarNames[i]
        
        if np.size(Weight) == 0:
            Weight = self.Weight
        if np.size(ClusterLabel_Weight) == 0:
            if np.size(self.ClusterLabel_Weight ) != 0:
                ClusterLabel_Weight = self.ClusterLabel_Weight        
            else:
                ClusterLabel_Weight = np.zeros((self.SOM_Dimension[0],self.SOM_Dimension[1])).astype(int)
        
            
        color_pallete = [['#1f77b4'], ['#ff7f0e'],[ '#2ca02c'], ['#d62728'], ['#9467bd'], ['#8c564b'], ['#e377c2'], ['#7f7f7f'], ['#bcbd22'],[ '#17becf']]

        
        hcoord = []
        vcoord = []
        colors_w = []
        CLabel = []
        for ii in range(self.SOM_Dimension[0]):
            for jj in range(self.SOM_Dimension[1]):
                if GridOpt == "hexagonal":
                    if np.mod(ii,2) == 0:
                        hcoord.append(jj+0.5)
                    else:
                        hcoord.append(jj)
                    vcoord.append(ii*np.sqrt(3)/2)
                else:
                    hcoord.append(jj)
                    vcoord.append(ii)
                    
                if np.size(ClusterLabel_Weight) == 1: # if it is False, the size is 1
                    colors_w.append("Grey")
                    CLabel.append(" ")
                else:
                    colors_w.append(color_pallete[ClusterLabel_Weight[ii,jj]])
                    CLabel.append(ClusterLabel_Weight[ii,jj])
                    
        if (self.n_var // ncol_subplot) == 0:
            ncol_subplot = self.n_var
            nrow_subplot = 1
        else:
            nrow_subplot = self.n_var // ncol_subplot + 1
        fig, axes = plt.subplots(nrow_subplot,ncol_subplot,
                                 figsize=(ncol_subplot * PlotSize*1.1,nrow_subplot * PlotSize))
           
        
        if nrow_subplot == 1:
            for i in range(self.n_var):
                temp_var = Weight[:,:,i]
                

    #            axes[ii,jj].set_aspect('equal')
    #            axes[i].set_xticks(np.arange(0,np.shape(ClusterLabel_Weight)[1]+1))
    #            axes[i].set_yticks(np.arange(0,np.shape(ClusterLabel_Weight)[0]+1))
                axes[i].axis('off')
                axes[i].set_title(titles[i],size=20,loc = "left")
    
                cmap = cm.ScalarMappable(None,colormap)
                norm = mapcolors.Normalize(vmin=temp_var.min(),vmax=temp_var.max())
                
                colors = []
                for ii in range(self.SOM_Dimension[0]):
                    for jj in range(self.SOM_Dimension[1]):
                        
                        colors.append(cmap.cmap(norm(temp_var[ii,jj])))
                
                # Add some coloured hexagons
                for x, y, c, c_w, cl in zip(hcoord, vcoord, colors,colors_w,CLabel):
                    
                    if GridOpt == "hexagonal":
                        hex = RegularPolygon((x, y), numVertices=6, radius=0.577, 
                                             orientation=np.radians(0), 
                                             facecolor=c, edgecolor='k')
                    elif GridOpt == "rectangular":        
                        hex = RegularPolygon((x, y), numVertices=4, radius=0.707, 
                                             orientation=np.radians(45), 
                                             facecolor=c, edgecolor = 'k')
                        
    #                if (x-1)>=0:
                        
                        
                    if np.max(ClusterLabel_Weight) !=0:
                        axes[i].text(x, y, cl+1, ha='center', va='center', size= 12,c = 'w',
                            path_effects=[pe.withStroke(linewidth=5, foreground=c_w[0].lower())])
                    axes[i].add_patch(hex)
                
                sm = plt.cm.ScalarMappable(cmap=colormap,norm=norm)
                sm._A = []
    #            plt.colorbar(sm,ticks=range(int(temp_var.min()),int(temp_var.max())+1))   
                plt.colorbar(sm,
                             ticks=np.arange(temp_var.min(),temp_var.max(),(temp_var.max() - temp_var.min())/10).round(1),
                             fraction=0.046, pad=0.04,ax =axes[i])   
#                plt.colorbar(sm,                             
#                             fraction=0.046, pad=0.04)  
    #            plt.clim(-0.5, 5.5)
                 
                axes[i].scatter(hcoord, vcoord, c=[c[0].lower() for c in colors_w],edgecolors = 'k') 
            
        else:
            for ia in range(nrow_subplot):
                for ja in range(ncol_subplot):
                    axes[ia,ja].axis('off')
                    
            for i in range(self.n_var):
                temp_var = Weight[:,:,i]
                
                ia = i // ncol_subplot
                ja = i % ncol_subplot
#                print(ia,ja)
    #            axes[ii,jj].set_aspect('equal')
    #            axes[i].set_xticks(np.arange(0,np.shape(ClusterLabel_Weight)[1]+1))
    #            axes[i].set_yticks(np.arange(0,np.shape(ClusterLabel_Weight)[0]+1))
                
                axes[ia,ja].set_title(titles[i],size=20,loc = "left")
    
                cmap = cm.ScalarMappable(None,colormap)
                norm = mapcolors.Normalize(vmin=temp_var.min(),vmax=temp_var.max())
                
                colors = []
                for ii in range(self.SOM_Dimension[0]):
                    for jj in range(self.SOM_Dimension[1]):
                        
                        colors.append(cmap.cmap(norm(temp_var[ii,jj])))
                
                # Add some coloured hexagons
                for x, y, c, c_w, cl in zip(hcoord, vcoord, colors,colors_w,CLabel):
                    
                    if GridOpt == "hexagonal":
                        hex = RegularPolygon((x, y), numVertices=6, radius=0.577, 
                                             orientation=np.radians(0), 
                                             facecolor=c, edgecolor='k')
                    elif GridOpt == "rectangular":        
                        hex = RegularPolygon((x, y), numVertices=4, radius=0.707, 
                                             orientation=np.radians(45), 
                                             facecolor=c, edgecolor = 'k')

                    if np.max(ClusterLabel_Weight) !=0:
                        axes[ia,ja].text(x, y, cl+1, ha='center', va='center', size= 12,c = 'w',
                            path_effects=[pe.withStroke(linewidth=5, foreground=c_w[0].lower())])
                    axes[ia,ja].add_patch(hex)
                
                sm = plt.cm.ScalarMappable(cmap=colormap,norm=norm)
                sm._A = []
    #            plt.colorbar(sm,ticks=range(int(temp_var.min()),int(temp_var.max())+1))   
                plt.colorbar(sm,
                             ticks=np.arange(temp_var.min(),temp_var.max(),(temp_var.max() - temp_var.min())/10).round(1),
                             fraction=0.046, pad=0.04,ax =axes[ia,ja] )   
    #            plt.clim(-0.5, 5.5)
                 
                axes[ia,ja].scatter(hcoord, vcoord, c=[c[0].lower() for c in colors_w],edgecolors = 'k') 


        plt.show()
        return fig, axes
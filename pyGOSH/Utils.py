# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:10:09 2022

@author: user
"""
# import numpy as np
# import pandas as pd
import os
import matplotlib.pyplot as plt

# def DynVis(T):
#     #Vogel's equation #mPa
#     A = -3.7188
#     B = 578.919
#     C = -137.546
    
#     po = A + B / (C + T)
    
#     return np.exp(po)
    
    
# def DensWater(T): 
#     #DIPPR105 -> kg/m3
#     A = 0.14395
#     B = 0.0112
#     C = 649.727
#     D = 0.05107
    
#     rho = A / (B ** (1+(1-T/C)**D))
    
#     return rho

# def DensWaterbeta(T):
#     #NIST
#     rH2O = 998.2071

#     beta = 0.0002
#     T0 = 293.15
    
#     rho = rH2O / (1+ beta*(T - T0))
    
#     return rho

# def DensWaterCivan(T):
#     #Civan2007
#     A =1.2538
#     B = -1.4496 * 10**3
#     C = 1.2971 * 10**5
#     T2 = T + 175
    
#     Reldev = A + B / T2 + C / T2**2
    
    
#     rho = (1 - np.exp(Reldev)) * 1065
    
#     return rho
 

# def KinVis_Cels(TCels):
#     #NIST
#     T0 = 273.15

#     TKelv = T0 + TCels
#     mu = DynVis(TKelv)
# #    rho= DensWater(T)
# #    rho= DensWaterbeta(T)
#     rho = DensWaterCivan(TCels)
    
#     mukPa = mu /10 **3
#     mukgm2 = mukPa 
#     nu = mukgm2/rho
    
#     return nu

def CheckMakeFolder(Folder):
    if not os.path.exists(Folder):
        os.makedirs(Folder)

def SavefigFolder(Folder,Filename,ext = 'png',dpi = 500,figobj =None):
    
    Path = Folder + '/' + Filename + '.' + ext
    if figobj == None:
        plt.tight_layout()
        if ext == 'pdf':
            plt.savefig(Path, bbox_inches='tight')
        else:
            plt.savefig(Path, bbox_inches='tight',dpi = dpi)
    else:
        plt.tight_layout()
        if ext == 'pdf':
            figobj.savefig(Path, bbox_inches='tight')
        else:
            figobj.savefig(Path, bbox_inches='tight',dpi = dpi)
        

    
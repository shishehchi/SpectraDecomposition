import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from typing import Dict
import matplotlib as mpl

from helpers.Classifier import Model_class as CLS
from helpers.Regressor_ms import Flux_estimator as REG_ms

#---------------------------------------
# Regressor for file_N (a file subtracted by median)
REG=REG_ms()
Classifer= CLS()

wl = np.load('data/wl_2Ao2.npy')
idx = np.where((wl>4000)&(wl<=7000))[0]

def Regressor(X,REG=REG,idx=idx):
    Xn=[]
    for k1 in range(len(X)):
        med= np.median(X[k1][idx])
        # std= np.std(X[k1][idx])
        Xn.append(X[k1]-med)
    Xn= np.array(Xn)
    return REG.predict(Xn)


#--------------------------------------------------------------
    
def PLOT_Single_Decompse_NII(Classifier, XC, n_r=1, Fig_save=0, Fig_name="fig-Decompose-BPT-01.png"):
    
    Xc=np.reshape(XC[n_r],(1,2208))
    Psf,Pagn,Pw,Pa,Pt = Classifier.predict(Xc)    
    wl = np.load('data/wl_2Ao2.npy')

    layout = [
        ["A", "A"],
        ["B", "C"],
        ["D", "D"]
    ]
    fig, axd = plt.subplot_mosaic(layout, figsize=(8,10))
    fig.subplots_adjust(hspace=100)
    axd['A'].plot(wl,XC[n_r],c='green',label='Combined')
    axd['A'].set_ylabel('Flux', fontsize=13)
    axd['A'].tick_params(axis='both', which='major', labelsize=13)
    axd['A'].set_xticks([])
    plt.xlim([3600,8200])
    axd['A'].legend()

    fig.subplots_adjust(hspace=.04)
    axd['B'].plot(wl,Pw[0][0]*Psf[0],c='blue',
                  label='SF (' + str(  np.round(100*Pw[0][0],2)) +'%)')
    axd['B'].tick_params(axis='both', which='major', labelsize=13)
    axd['B'].set_ylabel('Flux', fontsize=13)
    axd['B'].legend()
    #plt.xlim([6400,6800])
    fig.subplots_adjust(hspace=.04)
    axd['C'].plot(wl,(1-Pw[0][0])*Pagn[0],c='red',
                  label='AGN (' + str(  np.round(100*(1-Pw[0][0]),2)) +'%)')
    axd['C'].tick_params(axis='both', which='major', labelsize=13)
    axd['C'].set_ylabel('Flux', fontsize=13)
    axd['C'].legend()

    fig.subplots_adjust(hspace=.04)
    axd['D'].plot(wl,(1-Pw[0][0])*Pagn[0]+Pw[0][0]*Psf[0]-XC[n_r])
    axd['D'].set_ylabel('Flux (Combined-SF-AGN)', fontsize=13)
    axd['D'].tick_params(axis='both', which='major', labelsize=13)
    axd['D'].set_ylim([-1.5,1.5])
    plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=20)

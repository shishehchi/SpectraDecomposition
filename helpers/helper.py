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
    
def PLOT_Single_Decompse_NII(Classifier, XC, n_spec=1, Fig_save=0, Fig_name="fig-Decompose-BPT-01.png"):


    Xc=np.reshape(XC[n_spec],(1,2208))

    Psf,Pagn,Pw,Pa,Pt = Classifier.predict(Xc)

    Pw[Pw>1]=1
    Pw[Pw<0]=0


    Fs= Regressor(Psf)[0]
    Fa= Regressor(Pagn)[0]
    Fc= Regressor(Xc)[0]

    # Create layout
    layout = [
        ["C", "C", "D", "D"],
        ["R", "R", "D", "D"],    
        ["A", "F", "D", "D"],
        ["B", "B", "D", "D"]
    ]

    fig, axd = plt.subplot_mosaic(layout, figsize=(13,6))
    
    fig.subplots_adjust(hspace=.04)
    axd['C'].plot(wl,Xc[0], c='g', label='Combined')
    axd['C'].set_xticks([])
    axd['C'].set_ylabel('Flux', fontsize=13) 
    axd['C'].legend()

    fig.subplots_adjust(hspace=.04)
    axd['A'].plot(wl,Psf[0], c='b', label=str(np.round(100*Pw[0][0],2))+ '% SF')
    axd['A'].set_xticks([])
    axd['A'].set_ylabel('Normlized Flux', fontsize=19) 
    axd['A'].set_ylabel('Flux', fontsize=13) 
    axd['A'].legend()

    axd['R'].arrow(x=.5, y=1, dx=0, dy=-.5, width=.08,color='k') 
    axd['R'].set_xticks([])
    axd['R'].set_yticks([])
    axd['R'].patch.set_visible(False)
    axd['R'].axis('off')

    axd['F'].plot(wl,Pagn[0], c='r',label=str(np.round(100*(1-Pw[0][0]),2))+ '% AGN')
    plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=20)
    axd['F'].legend()
    axd['F'].set_ylabel('Flux', fontsize=13) 
    plt.xlim([3600,8200])

    
    fig.subplots_adjust(hspace=.04)
    axd['B'].plot(wl,(1-Pw[0][0])*Pagn[0]+Pw[0][0]*Psf[0]-XC[n_spec])
    axd['B'].set_ylabel('Flux (Combined-SF-AGN)', fontsize=13)
    axd['B'].tick_params(axis='both', which='major', labelsize=13)
    axd['B'].set_ylim([-1.5,1.5])
    plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=20)

    xk01= np.arange(-1.7,.47, .001)
    xk03= np.arange(-1.7,.04, .001)
    #s06=(-31.093+(-30.787+1.13581*(xx)+.27297*(xx)**2)*np.tanh(5.7409*(xx)))
    k01= (1.19+.61/(xk01-.47))
    k03 = (0.61 / (xk03 - 0.05)) + 1.3

    axd['D'].scatter(Fs[0],Fs[1], marker='*', c='b', s=200,label='SF' )
    axd['D'].scatter(Fa[0],Fa[1], marker='D', c='r', s=170,label='AGN' )
    axd['D'].scatter(Fc[0],Fc[1], marker='o', c='g', s=150,label='Combined' )
    axd['D'].legend()

    axd['D'].set_xlim([-1.6,1.3])
    axd['D'].set_ylim([-1.6,1.3])

    axd['D'].plot( xk03,k03,'--k', linewidth=2)
    axd['D'].plot( xk01,k01,'-k', linewidth=2)

    axd['D'].set_xlabel('log NII/H$\\alpha$', fontsize=15)
    axd['D'].set_ylabel('log OIII/H$\\beta$', fontsize=15)


    axd['D'].text(-.55, -1.1,'K03',fontsize=15)
    axd['D'].text(.25, -1,'K01',fontsize=15)

    # Adjusting the layout for better view

    plt.tight_layout()
    plt.show()

    if Fig_save:fig.savefig(Fig_name , format='png',bbox_inches='tight')


    print(n_spec)
    
    
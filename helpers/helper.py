import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from typing import Dict
import matplotlib as mpl

from Classifier import Model_class as CLS
from Regressor_ms import Flux_estimator as REG_ms
Classifier= CLS()



#---------------------------------------
# Regressor for file_N (a file subtracted by median)
REG=REG_ms()
Classifer= CLS()

wl = np.load('wl_2Ao2.npy')
idx = np.where((wl>4000)&(wl<=7000))[0]

def Regressor(X,REG=REG,idx=idx):
    Xn=[]
    for k1 in range(len(X)):
        med= np.median(X[k1][idx])
        std= np.std(X[k1][idx])
        Xn.append(X[k1]-med)
    Xn= np.array(Xn)
    return REG.predict(Xn)


#--------------------------------------------------------------


def Find_AGN_NII(FluxP): 
    n_agn=[]
    xBPT= FluxP[:,0]
    yBPT= FluxP[:,1]

    size=2

    for k1 in range(len(xBPT)):
        xx = xBPT[k1]
        yy = yBPT[k1]
        k03 = (0.61 / (xx - 0.05)) + 1.3
        k01= (1.19+.61/(xx-.47))
        if xx < 0:
            if yy < k03:
                n_agn.append(1)

            elif yy > k01:
                n_agn.append(-1)

            else:
                n_agn.append(0)
        else:
            if yy > k01 or xx > 0.4:

                n_agn.append(-1)
            else:
                n_agn.append(0)

    n_agn= np.array(n_agn)
    out= np.array([n_agn, np.array(xBPT), np.array(yBPT)])
    return out.T

#----------------------------------------------------------------

def Find_AGN_SII(FluxP): 
    n_agn=[]
    xBPT= FluxP[:,6]
    yBPT= FluxP[:,1]

    for k1 in range(len(xBPT)):
        xx = xBPT[k1]
        yy = yBPT[k1]
        
        if xx<0.32:
            if yy>= (0.72/(xx-0.32))+1.3:
                n_agn.append(-1)
            else:
                if yy < (0.72/(xx-0.32))+1.3:
                    n_agn.append(1)
            
        else:
            n_agn.append(-1)           

    n_agn= np.array(n_agn)
    out= np.array([n_agn, np.array(xBPT), np.array(yBPT)])
    return out.T


#---------------------------------------------------------------------

def PLOT_BPT_NII(X,Y,C,S=10):
    x_bpt=X
    y_bpt=Y
    n_class=C
    xk01= np.arange(-1.7,.47, .001)
    xk03= np.arange(-1.7,.04, .001)
    k01= (1.19+.61/(xk01-.47))
    k03 = (0.61 / (xk03 - 0.05)) + 1.3

    size=S

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x_bpt[n_class==1],y_bpt[n_class==1], marker='.', c='blue', s=size,label='SF' )
    ax.scatter(x_bpt[n_class==0],y_bpt[n_class==0], marker='.', c='green',s=size,label='Copm.')
    ax.scatter(x_bpt[n_class==-1],y_bpt[n_class==-1], marker='.', c='red', s=size, label='AGN')
    ax.legend()
    x_seyf= np.arange(-0.2,.6, .01)
    seyf= 1.05*x_seyf+0.45

    ax.set(xlim=(-1.4, .7))
    ax.set(ylim=(-1.2 ,1.2))
    ax.set_xlabel('log NII/H$\\alpha$',fontsize=30)
    ax.set_ylabel('log OIII/H$\\beta$',fontsize=30)
    ax.tick_params(labelsize=30,labelcolor="black")
    ax.text(-.5, -1.1,'K03',fontsize=25)
    ax.text(.2, -1,'K01',fontsize=25)
    ax.mpl.rcParams['axes.linewidth'] = 0.1


    plt.plot( xk03,k03,'--k', linewidth=3)
    plt.plot( xk01,k01,'-k', linewidth=3)

    plt.xlabel('log NII/H$\\alpha$', fontsize=30)
    plt.ylabel('log OIII/H$\\beta$', fontsize=30)
    
    plt.show()
    
    
#------------------------------------------------------------------

def PLOT_BPT_SII(X,Y,C,S=10):
    x_bpt=X
    y_bpt=Y
    n_class=C
    
    size=S
    xk01= np.arange(-2,.319999, .01)
    
    x_line= np.arange(0,1, .01)
    y_line =.5*x_line+.3
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x_bpt[n_class==1],y_bpt[n_class==1], marker='.', c='blue', s=size,label='SF' )
    ax.scatter(x_bpt[n_class==-1],y_bpt[n_class==-1], marker='.', c='red', s=size, label='AGN')
    ax.legend()

    yk01=  (0.72/(xk01-0.32))+1.3
    
    ax.plot(xk01,yk01,'--k',linewidth=3)
    
    ax.set(xlim=(-2, 2))
    ax.set(ylim=(-1.2 ,1.2))
    #ax.set_title(title)
    #ax.grid(b=False)
    ax.set_xlabel('log NII/H$\\alpha$',fontsize=30)
    ax.set_ylabel('log OIII/H$\\beta$',fontsize=30)
    ax.tick_params(labelsize=30,labelcolor="black")

    plt.xlabel('log SII/H$\\alpha$', fontsize=30)
    plt.ylabel('log OIII/H$\\beta$', fontsize=30)
    
    plt.show()
    
    
#------------------------------------------------------------------------


def PLOT_Single_Combine_NII(Xsf,Xagn,Wsf, XC,  n_spec=1 ,Fig_save=0, Fig_name="fig-Decompose-BPT-01.png"):

    Xsf=Xsf
    Xagn=Xagn
    Wsf=Wsf
    XC=XC
    n_spec=n_spec

    # Create layout
    layout = [
        ["A", "D"],
        ["B", "D"],
        ["R", "D"],
        ["C", "D"]
    ]

    fig, axd = plt.subplot_mosaic(layout, figsize=(13,6))

    fig.subplots_adjust(hspace=.04)
    axd['A'].plot(wl,Xsf[n_spec], c='b', label=str(np.round(100*(Wsf[n_spec][0]),2))+ '% SF' )
    axd['A'].set_xticks([])
    axd['A'].set_ylabel('Flux', fontsize=13) 
    axd['A'].legend()

    fig.subplots_adjust(hspace=.04)
    axd['B'].plot(wl,Xagn[n_spec], c='r',label=str(np.round(100*(1-Wsf[n_spec][0]),2))+ '% AGN')
    axd['B'].set_xticks([])
    axd['B'].set_ylabel('Flux', fontsize=13)
    axd['B'].annotate("HHH", xy=(2000, -1), xytext=(2000, -2), arrowprops=dict(arrowstyle="->"))
    axd['B'].legend()

    axd['R'].arrow(x=.5, y=1, dx=0, dy=-.5, width=.08,color='k') 

    axd['R'].set_xticks([])
    axd['R'].set_yticks([])
    axd['R'].patch.set_visible(False)
    axd['R'].axis('off')

    axd['C'].plot(wl,XC[n_spec], c='g',label='Combined')
    plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=20)
    axd['C'].set_ylabel('Flux', fontsize=13) 
    axd['C'].legend()
    plt.xlim([3600,8200])

    XX= np.array([Xsf[n_spec],Xagn[n_spec],XC[n_spec]])

    FluxP= Regressor(XX)

    pl =Find_AGN_NII(FluxP)

    xk01= np.arange(-1.7,.47, .001)
    xk03= np.arange(-1.7,.04, .001)
    #s06=(-31.093+(-30.787+1.13581*(xx)+.27297*(xx)**2)*np.tanh(5.7409*(xx)))
    k01= (1.19+.61/(xk01-.47))
    k03 = (0.61 / (xk03 - 0.05)) + 1.3

    axd['D'].scatter(pl[0,1],pl[0,2], marker='*', c='b', s=200,label='SF', alpha=.8)
    axd['D'].scatter(pl[1,1],pl[1,2],marker='D', c='r', s=120,label='AGN' , alpha=.8)
    axd['D'].scatter(pl[2,1],pl[2,2], marker='o', c='g', s=170,label='Combined', alpha=.8 )
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
    if Fig_save: fig.savefig(Fig_name, format='png',bbox_inches='tight')                            
    return n_spec
    
#=======================================================================================================




def PLOT_Single_Decompse_NII(X, n_spec=1, Fig_save=0, Fig_name="fig-Decompose-BPT-01.png"):
    XC=X
    n_spec=n_spec

    Xc=np.reshape(XC[n_spec],(1,2208))

    Psf,Pagn,Pw,Pa,Pt = Classifier.predict(Xc)

    Pw[Pw>1]=1
    Pw[Pw<0]=0


    Fs= Regressor(Psf)[0]
    Fa= Regressor(Pagn)[0]
    Fc= Regressor(Xc)[0]

    # Create layout
    layout = [
        ["C", "D"],
        ["R", "D"],    
        ["A", "D"],
        ["B", "D"]
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

    axd['B'].plot(wl,Pagn[0], c='r',label=str(np.round(100*(1-Pw[0][0]),2))+ '% AGN')
    plt.xlabel(r'$\lambda$ ($\AA$)', fontsize=20)
    axd['B'].legend()
    axd['B'].set_ylabel('Flux', fontsize=13) 
    plt.xlim([3600,8200])




    xk01= np.arange(-1.7,.47, .001)
    xk03= np.arange(-1.7,.04, .001)
    #s06=(-31.093+(-30.787+1.13581*(xx)+.27297*(xx)**2)*np.tanh(5.7409*(xx)))
    k01= (1.19+.61/(xk01-.47))
    k03 = (0.61 / (xk03 - 0.05)) + 1.3

    axd['D'].scatter(Fs[0],Fs[1], marker='*', c='b', s=200,label='SF' )
    axd['D'].scatter(Fa[0],Fa[1],marker='D', c='r', s=170,label='AGN' )
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
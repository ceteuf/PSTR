#%%
from pstr import PSTR
import pandas as pd 
import numpy as np



# %%

df = pd.read_csv("data/hansen99.csv")


new_pstr = PSTR(data=df,
                dep='inva', 
                indeps=list(df.columns[3:21]), 
                indeps_k=['vala','debta','cfa','sales'],
                tvars=[16],
                timeVar="year",
                indiVar="cusip")


# %%


# Evaluate the transition function.
#
# This function evaluate the transition function values. It is used by other functions in the package.
#
# If \code{vx} is a matrix, its row number must be equal to the length of \code{vc}.
#
# vx a vector or matrix of the transition variables.
# gamma the smoothness parameter.
# vc a vector of the location parameters, whose length is the number of switches in the transition function.
# return If vx is a vector, then a scalor retured, otherwise a vector.

def fTF(vx, gamma, vc):
    # depend on numpy as np
    if type(vc) == list :
        vc = np.array(vc).reshape(len(vc), 1)
    tmp = vx - vc 
    tmp = -1*np.prod(tmp,axis=1, keepdims=True)*gamma
    return 1 / (np.exp(tmp) + 1)




# %%
def EstPSTR(use, im=1, iq=None, par=None, useDelta=False, vLower=2, vUpper=2, method='L-BFGS-B'):
    # depend on numpy as np

    if type(use) != PSTR :
        raise ValueError("The argument 'use' is not an object of class 'PSTR'")

    ret = use 
    iT = use.i
    iN = use.t 

    # get the data here
    vY  = use.vY
    vYb = use.vYb
    mX  = use.mX
    mXb = use.mXb
    mK = use.mK
    ik = mK.shape[1]

    def ftmp(vx) :
        return vx - mean(vx)

    ret.imm = im # used in estimation

    ret.iq=iq


    if iq is not None : 
        if im < 1 :
            raise ValueError("The number of switches is invalid.")
        
        if type(iq) not in [float, int]:
            ret.iq = use.data.columns.get_loc(iq)

        if type(iq) == list :
            raise ValueError("Sorry! We only support the one transition variable case.")

        vQ = use.mQ[:,iq-1].reshape((use.mQ[:,iq-1].shape[0],1))
        mQ = vQ.transpose()

        def ResiduleSumSquare(vp) :
            # vp : variable "double" en R --> devra être donné par un liste / array a voir
            # vp[1] = log(gamma) or delta
            vg = fTF(vx=mQ, gamma= vp[0], vc =vp[1:len(vp)] )
            mXX = mK * vg 
            # spliting to take account for the panel structure : 3D array
            # iK arrays in which sub arrays have a shape of (iT, iN) 
            aXX = np.reshape(mXX, (ik,iT) + (iN,))










# %%

EstPSTR(new_pstr,im=1,iq=1,useDelta=True,par=[-0.462,0], vLower=4, vUpper=4)


# %%
test = pd.read_csv("C:/Users/Pierre/Desktop/reasearch_ideas/econometics_code/pstr/data/test.csv",sep=";")


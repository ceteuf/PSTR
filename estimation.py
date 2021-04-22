#%%
from pstr import PSTR
import pandas as pd 


# %%

df = pd.read_csv("data/hansen99.csv")

new_pstr = PSTR(data=df,
                dep='inva', 
                indeps=list(df.columns[3:21]), 
                indeps_k=['vala','debta','cfa','sales'],
                tvars=['vala'],
                timeVar="year",
                indiVar="cusip")



# %%
def EstPSTR(use, im=1, iq=NULL, par=NULL, useDelta=FALSE, vLower=2, vUpper=2, method='L-BFGS-B'):
    if type(use) != PSTR :
        raise ValueError("The argument 'use' is not an object of class 'PSTR'")

    ret = use 
    iT , iN = use.t , use.i 

    # get the data here
    vY , vYb = use.vY , use.vYb
    mX , mXb = use.mX , use.mXb
    mK = use.keepdims
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

        vQ = use.mQ[:,iq]
        mQ = # mQ = t(matrix(vQ,iT*iN,im)) 


        








# %%
import numpy as np

arr = df.year.values
arr.transpose()

# %%
from utils import MaClasse
import pandas as pd 
import numpy as np
from scipy.linalg import cholesky, inv, cho_solve, cho_factor
import mpmath as mp
np.set_printoptions(precision=8)
mp.mp.dps = 50
# %%
def checkIfElsInColDf(cols, lst):
    for el in lst:
        if not el in cols:
            return False
    return True


class PSTR(object):

    def __init__(self, data, dep, indeps, indeps_k, tvars, timeVar, indiVar):
        
        assert isinstance(data, pd.DataFrame), "data doit être un dataframe !"
        self.data = data

        assert isinstance(dep, str), "Dep doit une chaine de caractère"
        self.dep = dep

        assert isinstance(indeps, list) & checkIfElsInColDf(data.columns, indeps), "Indeps doit être un liste contenant des noms de colonnes valides !"
        self.indeps = indeps

        assert isinstance(indeps_k, list) & checkIfElsInColDf(data.columns, indeps_k), "Indeps_k doit être un liste contenant des noms de colonnes valides !"
        self.indeps_k = indeps_k

        assert isinstance(tvars, list) & checkIfElsInColDf(data.columns, tvars), "Tvars doit être un liste contenant des noms de colonnes valides !"
        self.tvars = tvars

        assert isinstance(timeVar, str)
        self.timeVar = timeVar

        assert isinstance(indiVar, str)
        self.indiVar = indiVar

        self.t_dim = data[timeVar].unique()
        self.i_dim = data[indiVar].unique()

        self.t = len(self.t_dim)
        self.i = len(self.i_dim)


        self.vY = data[dep].values # independante
        self.mX = data[indeps].values # dépendantes
        self.mK = data[indeps_k].values
        self.mQ = data[tvars].values


        coln = np.transpose([np.arange(1, self.i+1)] * self.t)
        coln = np.reshape(coln, coln.size)
        self.coln = coln
        
        vYb, mXb = np.zeros((self.i * self.t)), np.array([]).reshape(0, self.mX.shape[1])
        for ix in range(1, self.i+1, 1):
            tmp = (coln == ix)
            vYb[tmp] = self.vY[tmp] - np.mean(self.vY[tmp])
            val = np.transpose(np.transpose(self.mX[tmp]) - np.mean(np.transpose(self.mX), axis=1, keepdims=True))
            mXb = np.vstack([mXb, val])
        self.mXb = mXb
        self.vYb = vYb
        self.vY = self.vY[:, np.newaxis]


    def lmtest(self, vU, mW, mM, s2, mX2, invXX):
        df1 = mW.shape[1]
        df2 = (self.t*self.i) - df1 - self.i - self.mX.shape[1]
        #print(self.t, self.i, df1, self.mX.shape[1])
        mW2 = mM @ mW
        mXW2 = np.transpose(mX2) @ mW2
        #self.mXW2 = mXW2
        # S1 = ( crossprod(mW2) - t(mXW2) %*% invXX %*% mXW2 ) * s2
        #print(s2)
        S1 = (np.transpose(mW2) @ mW2 - (np.transpose(mXW2) @ invXX @ mXW2 )) * s2  # 
        self.S1 = S1
        ttmp = np.linalg.svd(S1)
        self.ttmp = ttmp
        # try:
        #     invS1 = np.linalg.inv(np.linalg.cholesky(S1))
        # except:
        #     ttmp = np.linalg.svd(S1)
        #     invS1 = ttmp.u 


    def lintest(self):
        mD = np.kron(np.eye(self.i),  np.repeat(1, self.t)[:, np.newaxis])
        mM = np.eye(self.i*self.t) - (mD @ np.transpose(mD)) / self.t
        mX2 = mM @ self.mX
        invXX = np.linalg.inv( np.transpose(mX2) @ mX2)
        self.invXX = invXX
        rr = np.transpose(self.mXb) @ self.mXb
        vv = np.transpose(self.mXb) @ self.vYb
        tmp = np.linalg.inv(rr) @ vv
        #check = np.transpose(self.mXb) @ self.vYb
        #check2 = np.linalg.inv(rr)
        # print(np.linalg.cond(np.transpose(self.mXb) @ self.mXb, p=2))
        tmp = tmp[:, np.newaxis]
        self.rr = rr
        self.tmp = tmp
        #self.check = check
        #self.check2 = check2
        # vU = np.reshape(self.vY - (self.mX @ tmp), (self.t, self.i))
        # vU_tmp = self.vY - (self.mX @ tmp)
        # vU = np.transpose(np.transpose(vU) - np.mean(np.transpose(vU), axis=1, keepdims=True)).flatten()[:, np.newaxis]
        # self.vU_tmp = vU_tmp
        # self.vU = vU
        # s2 = sum((vU - np.mean(vU, keepdims=True)) ** 2)   /(self.i*self.t)
        # print(s2)
        # coln = np.transpose([np.arange(1, self.i+1)] * self.t)
        # coln = np.reshape(coln, (coln.size, 1))
        # self.coln = coln
        # for qter in range(0, self.mQ.shape[1]):
        #    vQ = self.mQ[:, qter][:, np.newaxis]
        #    mW = self.mK*vQ
        #    self.lmtest(vU, mW, mM, s2, mX2, invXX)
        #    break

        #self.s2 = s2
        #invXX = np.linalg.inv(np.linalg.cholesky(mX2))
        #self.invXX = invXX

# %%

# %%
df = pd.read_csv("data/hansen99.csv") # .astype(np.float32)

# %%
new_pstr = PSTR(data=df,
                dep='inva', 
                indeps=list(df.columns[3:21]), 
                indeps_k=['vala','debta','cfa','sales'],
                tvars=['vala'],
                timeVar="year",
                indiVar="cusip")
# %%
new_pstr.invXX[0:3, 0:3]
# %%
new_pstr.lintest()
# %%
new_pstr.coln.shape
# %%
df.columns[3:21]
# %%
np.linalg.inv(new_pstr.rr)[0:3, 0:3]
# %%
np.linalg.solve(new_pstr.rr, np.eye(17))[0:3, 0:3]
# %%
c = np.linalg.inv(np.linalg.cholesky(new_pstr.rr))
np.dot(c.T,c)[0:3, 0:3]
# %%
A = new_pstr.rr
c, low = cho_factor(A)

x = cho_solve((cho_factor(A), False), np.eye(17))[0:3, 0:3]

# %%
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# import R's "base" package
base = importr('float')
# %%
base.chol2inv(base.chol(A))[0:2,0:2]
# %%
np.linalg.cond(new_pstr.rr, p=2)
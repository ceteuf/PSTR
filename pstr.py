# %%
from utils import MaClasse
import pandas as pd 
import numpy as np
from scipy.linalg import cholesky, inv
# %%
def checkIfElsInColDf(cols, lst):
    for el in lst:
        if not el in cols:
            return False
    return True


class PSTR(object):

    def __init__(self, data, dep, indeps, indeps_k, tvars, timeVar, indiVar):

        # ranger data : groupby indiv puis croissant par années

        assert isinstance(data, pd.DataFrame), "data doit être un dataframe !"
        self.data = data

        assert isinstance(dep, str), "Dep doit une chaine de caractère"
        self.dep = dep

        assert isinstance(indeps, list) & checkIfElsInColDf(data.columns, indeps), "Indeps doit être un liste contenant des noms de colonnes valides !"
        self.indeps = indeps

        assert isinstance(indeps_k, list) & checkIfElsInColDf(data.columns, indeps_k), "Indeps_k doit être un liste contenant des noms de colonnes valides !"
        self.indeps_k = indeps_k

        # correction : tvars est une liste de numéro de colonnes des variables de transitions potentielles
        # assert isinstance(tvars, list) & checkIfElsInColDf(data.columns, tvars), "Tvars doit être un liste contenant des noms de colonnes valides !"
        # self.tvars = tvars

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
        self.mQ_name = list(data.iloc[:, tvars])
        self.mQ = data[self.mQ_name].values
        # self.mQ = data[tvars].values


        coln = np.transpose([np.arange(1, self.i+1)] * self.t)
        coln = np.reshape(coln, coln.size)
        self.coln = coln
        
        vYb = np.zeros((self.i * self.t))
        # mXb = np.array([]).reshape(0, self.mX.shape[1])
        mXb = np.zeros((self.i*self.t, self.mX.shape[1]))
        for ix in range(1, self.i+1, 1):
            tmp = (coln == ix)
            vYb[tmp] = self.vY[tmp] - np.mean(self.vY[tmp])
            # temp_val = self.mX[tmp, :] - np.mean(np.transpose(self.mX[tmp, :]), axis=0, keepdims=True)
            #tmptmp1 = np.mean(np.transpose(self.mX[tmp,:]), axis=1, keepdims=True)
            val = np.transpose(np.transpose(self.mX[tmp,:]) - np.mean(np.transpose(self.mX[tmp,:]), axis=1, keepdims=True)  )
            mXb[tmp, :] = val
        
        self.mXb = mXb
        self.vYb = vYb
        self.vY = self.vY[:, np.newaxis]





# %%

# %%
df = pd.read_csv("data/hansen99.csv")
# %%

new_pstr = PSTR(data=df,
                dep='inva', 
                indeps=list(df.columns[3:21]), 
                indeps_k=['vala','debta','cfa','sales'],
                tvars=[16],
                timeVar="year",
                indiVar="cusip")
# %%

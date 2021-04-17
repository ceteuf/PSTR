# %%
from utils import MaClasse
import pandas as pd 

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



# %%

# %%
df = pd.read_csv("hansen99.csv")
# %%

new_pstr = PSTR(data=df,
                dep='inva', 
                indeps=["dt_75", "dt_76", "dt_77", "dt_78"], 
                indeps_k=['vala','debta','cfa','sales'],
                tvars=['vala'],
                timeVar="year",
                indiVar="cusip")
# %%
new_pstr.i
# %%
df["year"]
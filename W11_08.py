#%%
# Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

# %%
# Read data
df = pd.read_csv('week_11_nonlinear_data.csv')
# %%
df.info()
# %%
df
# %%
df_train = df.loc[:, ['x','y']].copy()
df_train
# %%
mod_00 = smf.ols(formula='y ~ 1', data=df_train).fit()

# %%
mod_00.params
# %%
mod_00.bse
# %%
mod_00.conf_int().\
rename(columns={0: 'ci_lwr', 1: 'ci_upr'})
# %%
def my_coefplot(mod, figsize_use=(10,4)):
    fig, ax = plt.subplots(figsize=figsize_use)
    ax.errorbar(y=mod.params.index,
                x=mod.params,
                xerr = 2 * mod.bse,
                fmt = 'o', color='k', ecolor='k', elinewidth=2, ms=10)
    ax.axvline(x=0, linestyle='--', linewidth=3.5, color='grey')
    ax.set_xlabel('coefficient value')
    plt.show()

my_coefplot(mod_00)
# %%
mod_00.rsquared
# %%
mod_00.resid
# %%
mod_00.resid ** 2
# %%
np.sqrt((mod_00.resid ** 2).mean())
# %%
mod_01 = smf.ols(formula = 'y~x', data=df_train).fit()
my_coefplot(mod_01)
# %%
sns.relplot(data = df_train, x='x', y='y')
plt.show()
# %%
mod_01.pvalues < 0.05
# %%
sns.lmplot(data=df_train, x='x', y='y')
plt.show()
# %%
mod_01.rsquared

# %%
mod_01.rsquared > mod_00.rsquared
# %%
np.sqrt((mod_01.resid ** 2).mean())
# %%
np.sqrt((mod_01.resid ** 2).mean()) > np.sqrt((mod_00.resid ** 2).mean())

# %%
print(mod_01.summary())
# %%
np.sqrt((mod_01.resid ** 2).mean()) < np.sqrt((mod_00.resid ** 2).mean())
# %%
mod_02 = smf.ols(formula='y ~ x + np.power(x, 2)', data=df_train).fit()
mod_02.params
# %%
my_coefplot(mod_02)
# %%
mod_02.rsquared
# %%
mod_01.rsquared
# %%
mod_03 = smf.ols(formula='y ~ x + np.power(x,2) + np.power(x,3)', data = df_train).fit()
my_coefplot(mod_03)
# %%
mod_03.rsquared
# %%
mod_03.rsquared > mod_01.rsquared
# %%
np.sqrt((mod_03.resid ** 2).mean()) < np.sqrt((mod_01.resid ** 2).mean())
# %%
np.sqrt((mod_03.resid ** 2).mean())
# %%
np.sqrt((mod_01.resid ** 2).mean())
# %%
sns.lmplot(data = df_train, x='x', y='y', order = 3)
plt.show()
# %%

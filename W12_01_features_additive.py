## Working with LINEAR MODELS with ADDITIVE features
#%%
## import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.formula.api as smf


# %%
##　Linear Additive Features
b0 = -0.25
b1 = 1.95
b2 = 0.2


# %%
N = 35 ### number of observation
rg = np.random.default_rng(2100)
# Call the random number generator for 2 inputs

input_df = pd.DataFrame({'x1':rg.normal(loc=0, scale=1, size = N),
                        'x2':rg.normal(loc=0, scale=1, size = N)})
# %%
input_df.info()
# %%
input_df.head()
# %%
df = input_df.copy()
#%%
df['trend'] = b0 + b1 * df.x1 + b2 * df.x2
# %%
my_sigma = 0.85
df['y'] = rg.normal(loc=df.trend, scale=my_sigma, size=df.shape[0])
df.info()
# %%
df.trend

# %%
df.head()


# %%
sns.relplot(data=df, x='x1', y='y', kind='scatter')
plt.show()
# %%
sns.relplot(data = df, x='x2', y='y', kind='scatter')
plt.show()
# %%
sns.lmplot(data=df, x='x1', y='y')
plt.show()

# %%
sns.lmplot(data=df, x='x2', y='y')
plt.show()
# %%
print(b2)
# %%
fit_x1 = smf.ols(formula='y ~ x1', data=df).fit()
# %%
fit_x1.params
# %%
fit_x1.bse
# %%
fit_x1.pvalues

# %%
fit_x1.pvalues<0.05
# %%
fit_x1.conf_int()
# %%
fit_x1.conf_int().rename(columns={0:'conf_lwr', 1:'conf_upr'})
# %%
fit_x2 = smf.ols(formula='y ~ x2', data=df).fit()
#%%
fit_x2.params

# %%
fit_x2.bse
# %%
fit_x2.pvalues
# %%
fit_x2.conf_int()
# %%
fit_x2.conf_int().rename(columns = {0: 'conf_lwr', 1:'conf_upr'})
# %%
def my_coefplot(mod, figsize_use=(10,4)):
    fig, ax = plt.subplots(figsize=figsize_use)

    ax.errorbar(y=mod.params.index,
                x=mod.params,
                xerr = 2 * mod.bse,
                fmt='o', color='k', ecolor='k', elinewidth=2, ms=10)
    ax.axvline(x=0, linestyle='--', linewidth=3.5, color='grey')
    ax.set_xlabel('coefficient value')
    plt.show()

# %%
my_coefplot(fit_x1)
# %%
my_coefplot(fit_x2)
# %%

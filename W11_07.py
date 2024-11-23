#%%
## Regression performance Metrics

## Import the Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

## Read the data
df = pd.read_csv('week_11_linear_data.csv')
df



# %%
lm_fit = smf.ols(formula = 'y ~ x', data=df).fit()
lm_fit.params
# %%
lm_fit.bse

# %%
lm_fit.pvalues
# %%
lm_fit.conf_int().\
rename(columns={0: 'ci_lwr', 1:'ci_upr'})
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
# %%
my_coefplot(lm_fit, figsize_use=(8,4))
# %%
lm_fit.fittedvalues
# %%
df_copy = df.loc[:, ['x', 'y']].copy()
df_copy
# %%
df_copy['fitted'] = lm_fit.fittedvalues
df_copy


# %%
sns.relplot(data=df_copy, x='x', y='fitted')
plt.show()
# %%
fig, ax = plt.subplots()
sns.scatterplot(data = df_copy, x='x', y='fitted', ax=ax)
sns.scatterplot(data = df_copy, x='x', y='y', ax=ax)
plt.show()
# %%
df_copy
# %%
sns.relplot(data=df_copy, x='y', y='fitted')
plt.show()
# %%
sns.set_style('whitegrid')
fig, ax = plt.subplots()
sns.scatterplot(data = df_copy, x='y', y='fitted',s=150, ax=ax)
sns.lineplot(data = df_copy, x='y', y='y', color='red', ax=ax)
plt.show()
# %%
df_copy.loc[:,['y','fitted']]

# %%
df_copy.loc[:,['y','fitted']].corr()
# %%
df_copy.loc[:,['y', 'fitted']].corr(numeric_only=True)
# %%
df_copy.loc[:,['y','fitted']].corr().iloc[0,1]
# %%
df_copy.loc[:,['y','fitted']].corr().iloc[0,1]**2
# %%
lm_fit.rsquared
# %%
df_copy
# %%
df_copy['errors'] = df_copy.y - df_copy.fitted
# %%
df_copy
# %%
sns.relplot(data=df_copy, x='x', y='errors')
plt.show()
# %%
lm_fit.resid
# %%
df_copy['residuals'] = lm_fit.resid
df_copy
# %%
lm_fit.resid ** 2
# %%
(lm_fit.resid ** 2).sum()
# %%
(lm_fit.resid ** 2).mean()
# %%
np.sqrt((lm_fit.resid ** 2).mean())
# %%
lm_fit.resid
# %%
np.abs(lm_fit.resid)
# %%
(np.abs(lm_fit.resid)).mean()
# %%

#%%
#Introduction to fitting LOGISTIC REGRESSION with STATSMODELS
## Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
#%%
# Read Data
df = pd.read_csv('week_12_intro_binary_classification.csv')
df.info()

# %%
df.nunique()
# %%
df.y.value_counts()
# %%
sns.catplot(data=df, x='y', kind='count')
plt.show()
# %%
df.describe()
# %%
df.y.mean()
# %%
df.y.value_counts()
# %%
df.y.value_counts(normalize=True)
# %%
sns.displot(data=df, x='x', kind='hist')
plt.show()
# %%
sns.catplot(data=df, x='y', y='x', kind='box')
plt.show()
# %%
sns.catplot(data=df, x='y', y='x', kind='violin')
plt.show()
# %%
sns.catplot(data=df, x='y', y='x', kind='point', join=False)
plt.show()
# %%
fit_glm = smf.logit(formula='y ~ x', data = df).fit()
# %%
fit_ols = smf.ols(formula='y ~ x', data=df).fit()
# %%
fit_glm.params
# %%
fit_glm.bse
# %%
fit_glm.pvalues
# %%
fit_glm.conf_int().\
rename(columns={0: 'conf_lwr', 1:'conf_upr'})
# %%
fit_glm.pvalues<0.05
# %%
def my_coefplot(mod, figsize_use=(10,4)):
    fit, ax = plt.subplots(figsize=figsize_use)
    ax.errorbar(y=mod.params.index,
                x=mod.params,
                xerr=2 * mod.bse,
                fmt='o', color='k',ecolor='k', elinewidth=2, ms=10 )
    ax.axvline(x=0, linestyle='--', linewidth=3.5, color='grey')
    ax.set_xlabel('coefficient value')
    plt.show()

my_coefplot(fit_glm)
# %%
df.describe()
# %%
input_grid = pd.DataFrame({'x':np.linspace(-5, 5, num=251)})
input_grid

# %%
df_viz = input_grid.copy()
# %%
df_viz['pred_probability'] = fit_glm.predict(input_grid)
df_viz
# %%
df_viz.describe()
# %%
sns.relplot(data=df_viz, x='x', y='pred_probability', kind='line')
plt.show()

# %%
fit_ols.params
# %%
fit_ols.bse
# %%
fit_ols.pvalues
# %%
my_coefplot(fit_ols)
# %%
df_viz['pred_from_ols'] = fit_ols.predict(input_grid)
sns.relplot(data=df_viz, x='x', y='pred_from_ols', kind='line')
plt.show()
# %%
sns.lmplot(data = df, x='x', y='y')
plt.show()
# %%
sns.lmplot(data=df, x='x', y='y', logistic=True)
plt.show()
# %%

#%%
## import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

# Read data
df = pd.read_csv('week_12_categorical_input.csv')
df.info()
# %%
df.nunique()
# %%
df.isna().sum()
# %%
sns.displot(data=df, x='y', kind='hist')
plt.show()
# %%
sns.catplot(data=df, x='x', kind='count')
plt.show()
# %%
sns.catplot(data=df, x='x', y='y', kind='box')
plt.show()
# %%
sns.catplot(data=df, x='x', y='y', kind='point', join=False)
plt.show()
# %%
df
# %%
fit_a = smf.ols(formula='y~x', data=df).fit()
# %%
fit_a.params
# %%
fit_a.params.size
# %%
df.x.nunique()

# %%
df.x.nunique() - 1

# %%
fit_a.params
# %%
df.groupby('x').aggregate(avg_y = ('y', 'mean')).reset_index()
# %%
avg_y_at_xA = df.loc[df.x == 'A', 'y'].mean()
avg_y_at_xA
# %%
df_summary = df.groupby('x').\
aggregate(avg_y = ('y', 'mean')).\
reset_index()

# %%
df_summary
# %%
df_summary['avg_at_xA'] = avg_y_at_xA
df_summary
# %%
df_summary['relative_difference_to_xA'] = df_summary.avg_y - df_summary.avg_at_xA
# %%
df_summary.round(3)
# %%
fit_a.params
# %%
fit_a.pvalues
# %%
fit_a.pvalues<0.05
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
# %%
my_coefplot(fit_a)
# %%
input_grid = pd.DataFrame({'x':df.x.unique()})
input_grid.sort_values('x', inplace=True, ignore_index=True)
input_grid
# %%
fit_a.predict(input_grid)
# %%
df_summary['pred_from_dummies'] = fit_a.predict(input_grid)
df_summary
# %%
fit_b = smf.ols(formula='y ~ x - 1', data=df).fit()
fit_b.params


# %%
df_summary['onehot_coefs'] = fit_b.params
# %%
df_summary
# %%
fit_b.params.reset_index()
# %%
df_summary['onehot_coefs'] = pd.Series(fit_b.params.reset_index(drop=True), index=df_summary.index)
# %%
df_summary
# %%
df_summary['pred_from_conehot'] = fit_b.predict(input_grid)
df_summary
# %%
my_coefplot(fit_b)
# %%
fit_b.pvalues<0.05
# %%

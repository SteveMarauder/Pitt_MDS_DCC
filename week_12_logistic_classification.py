#%%
# Making CLASSIFICATIONS with LOGISTIC REGRESSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
#%%
## Read data
df = pd.read_csv('week_12_intro_binary_classification.csv')
df.info()
# %%
df.nunique()
# %%
df.y.value_counts()
# %%
fit_glm = smf.logit(formula='y ~ x', data=df).fit()

# %%
fit_glm.params
# %%
fit_glm.bse
# %%
fit_glm.pvalues
# %%
def my_coefplot(mod, figsize_use=(10,4)):
    fig, ax = plt.subplots(figsize=figsize_use)
    ax.errorbar(y = mod.params.index,
                x = mod.params,
                xerr = 2 * mod.bse,
                fmt='o', color='k', ecolor='k', elinewidth=2, ms=10)
    ax.axvline(x=0, linestyle='--', linewidth=3.5, color='grey')
    ax.set_xlabel('coefficient value')
    plt.show()
# %%
my_coefplot(fit_glm)
# %%
df.describe()
# %%
dfviz = pd.DataFrame({'x': np.linspace(-2.75, 2.75, num=101)})
dfviz['pred_probability'] = fit_glm.predict(dfviz)
dfviz
# %%
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(data=df, x='x', y='y', s=150, alpha=0.5)
ax.axhline(y=0.25, color='green', linestyle='--', linewidth=2)
ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=2)
ax.axhline(y=0.75, color='orange', linestyle='--', linewidth=2)
ax.axvline(x=-1, color='black', linestyle=':')
ax.axvline(x=1, color='black', linestyle=':')
sns.lineplot(data=dfviz, x='x', y='pred_probability', color='k')
plt.show()
# %%
df_copy = df.copy()

# %%
df
# %%
df_copy['x_bin'] = pd.cut(df.x, [df.x.min(), -1, 1, df.x.max()], include_lowest=True, ordered=True)
# %%
df_copy.info()
# %%
df_copy.nunique()
# %%
df_copy.x_bin.value_counts()
# %%
sns.catplot(data=df_copy, x='x_bin', kind='count')
plt.show()
# %%
df_copy.groupby('x_bin').\
aggregate(num_rows = ('y', 'size'),
          num_events = ('y', 'sum'),
          prop_events = ('y', 'mean')).\
reset_index()
# %%
df_copy.head()
# %%
df_copy.shape
# %%
df_copy['pred_probability'] = fit_glm.predict(df)
# %%
df_copy
# %%
df_copy.pred_probability > 0.5
# %%
np.where(df_copy.pred_probability > 0.5, 1, 0)
# %%
df_copy
# %%
df_copy['pred_class'] = np.where(df_copy.pred_probability>0.5, 1, 0)
# %%
df_copy
# %%
df_copy.pred_class.value_counts()
# %%
sns.catplot(data=df_copy, x='pred_class', kind='count')
plt.show()
# %%

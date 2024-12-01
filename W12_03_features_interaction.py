#%%
# Working with LINEAR MODELS with INTERACTION features
#%%
## Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

#%%
df = pd.read_csv('linear_additive_example.csv')
df.info()

# %%
df.head()
# %%
fit_a = smf.ols(formula='y ~ x1 + x2 + x1 * x2', data=df).fit()
fit_a.params
# %%
fit_b = smf.ols(formula= 'y ~ x1 + x2 + x1:x2', data=df).fit()
fit_b.params
# %%
fit_a.bse

# %%
fit_b.bse
# %%
fit_a.pvalues
# %%
fit_b.pvalues
# %%
fit_c = smf.ols(formula='y ~ x1 * x2', data=df).fit()
# %%
fit_c.params
# %%
fit_c.bse
# %%
fit_c.pvalues
# %%
fit_d = smf.ols(formula='y ~ x1:x2', data=df).fit()
fit_d.params
# %%
df.x1.head() ** 2
# %%
np.power(df.x1.head().to_numpy(),2)
# %%
fit_e = smf.ols(formula='y ~ x1 ** 2 + x2 ** 2', data=df).fit()
# %%
fit_e.params
# %%
fit_f = smf.ols(formula='y ~ x1 + x2', data=df).fit()
# %%
fit_f.params
# %%
fit_g = smf.ols(formula='y~np.power(x1,2) + np.power(x2,2)', data=df).fit()
# %%
fit_g.params
# %%
fit_h = smf.ols(formula='y ~ x1 + np.power(x1,2) + x2 + np.power(x2, 2)', data=df).fit()
# %%
fit_h.params
# %%
fit_i = smf.ols(formula='y ~ (x1 + x2)**2', data=df).fit()
# %%
fit_i.params
# %%
fit_i = smf.ols(formula='y ~ (x1 + x2)', data=df).fit()
fit_i.params
# %%
fit_j = smf.ols(formula = 'y ~ (x1 + np.power(x1,2) + x2 + np.power(x2,2))**2', data=df).fit()
# %%
fit_j.params
# %%
fit_k = smf.ols(formula = 'y ~ (x1 + np.power(x1,2)) * (x2 + np.power(x2,2))', data=df).fit()
fit_k.params
# %%
input_grid = pd.DataFrame([(x1, x2) for x1 in np.linspace(df.x1.min(), df.x1.max(), num=101)
                                    for x2 in np.linspace(df.x2.min(), df.x2.max(), num=9)],
                            columns = ['x1', 'x2'])
input_grid.nunique()
# %%
viz_grid = input_grid.copy()

# %%
viz_grid['pred'] = fit_k.predict(input_grid)
# %%
input_grid
# %%
sns.relplot(data=viz_grid,
            x='x1', y='pred', kind='line',
            hue='x2', palette='coolwarm',
            estimator=None, units='x2')
plt.show()
# %%

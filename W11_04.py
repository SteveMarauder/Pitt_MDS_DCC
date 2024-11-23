### Fitting Linear Models to predict NON-LINEAR output to input relationships
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf


# %%
df = pd.read_csv('week_11_nonlinear_data.csv')
df.info()
# %%
df
# %%
lm_fit = smf.ols(formula = 'y ~ np.sin(x)' , data=df).fit()
# we would use np.sin(x) to generate non-linear function
type(lm_fit)
# %%
lm_fit.summary()
# %%

lm_fit.params

# %%
lm_fit.bse
# %%
lm_fit.pvalues
# %%
lm_fit.pvalues < 0.05
# %%
lm_fit.conf_int()
# %%
lm_fit.conf_int().\
rename(columns={0: 'ci_lwr', 1:'ci_upr'})

# %%
fig, ax = plt.subplots()
ax.errorbar(y = lm_fit.params.index,
            x = lm_fit.params,
            xerr = 2 * lm_fit.bse,
            fmt='o', color='k', ecolor='k', elinewidth=2, ms=10)

ax.axvline(x=0, linestyle='--', linewidth=4, color='grey')
ax.set_xlabel('coefficient value')
plt.show()
# %%

fig, ax = plt.subplots()
ax.errorbar(y = lm_fit.params.index,
            x = lm_fit.params,
            xerr = 2 * lm_fit.bse,
            fmt='o', color='k', ecolor='k', elinewidth=2, ms=10)

ax.scatter(y=lm_fit.params.index,
           x = [np.unique(df.true_intercept), np.unique(df.true_slope)],
           color='red', s=125)

ax.axvline(x=0, linestyle='--', linewidth=4, color='grey')
ax.set_xlabel('coefficient value')
plt.show()

# %%
df
# %%

#%%
## CMPINF 2100
### Fit a linear model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
import statsmodels.formula.api as smf
# %%
df = pd.read_csv('week_11_linear_data.csv')
# %%
df.info()
# %%
lm_fit = smf.ols(formula='y ~ x' , data=df).fit()
# %%
type(lm_fit)
# %%
dir(lm_fit)
# %%
lm_fit.summary()
# %%
print(lm_fit.summary())
# %%
lm_fit.params

# %%
lm_fit.params - 2 * lm_fit.bse
# %%
lm_fit.params + 2 * lm_fit.bse
# %%
lm_fit.conf_int()
# %%
type(lm_fit.conf_int())
# %%
lm_fit.conf_int().columns
# %%
lm_fit.conf_int().\
    rename(columns={0:'ci_lwr', 1:'ci_upr'})
# %%
coef_fit_info = lm_fit.conf_int().\
    rename(columns={0: 'ci_lwr', 1:'ci_upr'})
coef_fit_info
# %%
coef_fit_info['estimate'] = lm_fit.params

# %%
coef_fit_info['estimate_se'] = lm_fit.bse
# %%
coef_fit_info
# %%
coef_fit_info.index
# %%
lm_fit.pvalues
# %%
lm_fit.pvalues < 0.05
# %%
coef_fit_info
# %%
lm_fit.params

# %%
lm_fit.bse
# %%
lm_fit.conf_int()
# %%
fig, ax = plt.subplots()

### use the errorbar method to show the estimates as markers and CI as errorbars.
### create a HORIZONTAL errorbar
ax.errorbar(y = coef_fit_info.index,
            x = coef_fit_info.estimate,
            xerr = 2 * coef_fit_info.estimate_se,
            fmt='o', color = 'k', ecolor='k',elinewidth=2, ms=10)
### INCLUDE A VERTICAL REFERENCE LINE at 0
ax.axvline(x=0, linestyle='--', linewidth=4, color='grey')

### set the axis labels
ax.set_xlabel('coefficient value')
ax.set_ylabel('coefficient name')
plt.show()
# %%
coef_fit_info = lm_fit.conf_int().\
    rename(columns={0: 'ci_lwr', 1:'ci_upr'})
coef_fit_info
# %%
lm_fit.pvalues


# %%
fig, ax = plt.subplots()

### the 95% CI approximation around the estimate
ax.errorbar(y = lm_fit.params.index,
            x = lm_fit.params,
            xerr = 2 * lm_fit.bse,
            fmt = 'o', color ='k', ecolor ='k', elinewidth=2, ms=10)

### reference line at 0
ax.axvline(x=0, linestyle='--', linewidth=4, color='grey')

### axis labels
ax.set_xlabel('coefficient value')

plt.show()
# %%
lm_fit.pvalues
# %%

fig, ax = plt.subplots()
### the 95% CI approximation around the estimate
ax.errorbar(y= lm_fit.params.index,
            x = lm_fit.params,
            xerr = 2 * lm_fit.bse,
            fmt = 'o', color = 'k', ecolor = 'k', elinewidth=2, ms=10)

### reference line at 0
ax.axvline(x=0, linestyle='--', linewidth=4, color='grey')

### axis labels
ax.set_xlabel('coefficient value')

plt.show()
# %%
lm_fit.conf_int()
# %%
coef_fit_info
# %%
coef_fit_info['lwr_err'] = coef_fit_info.estimate - coef_fit_info.ci_lwr
coef_fit_info['upr_err'] = coef_fit_info.ci_upr - coef_fit_info.estimate
coef_fit_info
# %%
fig, ax = plt.subplots()
ax.errorbar( y = coef_fit_info.index,
             x = coef_fit_info.estimate,
             xerr = [coef_fit_info.lwr_err, coef_fit_info.upr_err],
            fmt = 'o', color='k', ecolor='k', elinewidth=3, ms=10)
ax.axvline(x=0, linestyle='--', linewidth=5, color='grey')
ax.set_xlabel('coefficient value')
plt.show()
# %%
df
# %%
fig, ax = plt.subplots()
ax.errorbar( y = coef_fit_info.index,
             x = coef_fit_info.estimate,
             xerr = [coef_fit_info.lwr_err, coef_fit_info.upr_err],
            fmt = 'o', color='k', ecolor='k', elinewidth=3, ms=10)

### ADD IN the TRUE COEFFICIENTS as MARKERS that generated the data
ax.scatter(y=lm_fit.params.index,
           x=[np.unique(df.true_intercept), np.unique(df.true_slope)],
          color = 'red', s=125)


ax.axvline(x=0, linestyle='--', linewidth=5, color='grey')
ax.set_xlabel('coefficient value')
plt.show()
# %%

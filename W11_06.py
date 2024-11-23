## Making predictions with linear models
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
#%%
df = pd.read_csv('week_11_nonlinear_data.csv')
df.info()
# %%
lm_fit = smf.ols(formula='y ~ np.sin(x)', data=df).fit()
lm_fit.params

# %%
lm_fit.bse
# %%
lm_fit.pvalues
# %%
lm_fit.conf_int().\
rename(columns={0: 'ci_lwr', 1:'ci_upr'})
# %%
df_viz = pd.DataFrame({'x': np.linspace(df.x.min()-0.1, df.x.max()+0.1, num=101)})
df_viz
# %%
df
# %%

predictions = lm_fit.get_prediction(df_viz)
# %%
lm_pred_summary = predictions.summary_frame()
lm_pred_summary
# %%
fig, ax = plt.subplots()

# prediction interval

ax.fill_between(df_viz.x,
                lm_pred_summary.obs_ci_lower, lm_pred_summary.obs_ci_upper,
                facecolor='orange', alpha=0.75, edgecolor='orange')

# confidence interval

ax.fill_between(df_viz.x,
                lm_pred_summary.mean_ci_lower, lm_pred_summary.mean_ci_upper,
                facecolor='grey', edgecolor='grey')


# trend 
ax.plot(df_viz.x, lm_pred_summary['mean'], color='k', linewidth=1.1)

# include the training data
ax.scatter(df.x, df.y,color='k')


# set the labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# show the plot
plt.show()

# %%

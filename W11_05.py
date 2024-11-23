##　Making Prediction with linear models
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

# %%
df = pd.read_csv('week_11_linear_data.csv')
df
# %%
lm_fit = smf.ols(formula='y ~ x', data=df).fit()


# %%
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
lm_pred = lm_fit.predict(df_viz)
# %%
type(lm_pred)
# %%
lm_pred
# %%
lm_pred.size
# %%
df_viz_copy = df_viz.copy()
df_viz_copy['pred_trend'] = lm_pred
df_viz_copy
# %%
fig, ax =plt.subplots()
ax.plot(df_viz_copy.x, df_viz_copy.pred_trend, color='b')

ax.set_xlabel('x')
ax.set_ylabel('trend')
plt.show()
# %%
fig, ax =plt.subplots()
ax.plot(df_viz_copy.x, df_viz_copy.pred_trend, color='b')

ax.scatter(df.x, df.y, color='k')

ax.set_xlabel('x')
ax.set_ylabel('trend')
plt.show()
# %%
df

# %%
my_intercept = np.unique(df.true_intercept)
my_slope = np.unique(df.true_slope)



# %%
fig, ax = plt.subplots()

ax.plot(df_viz_copy.x, df_viz_copy.pred_trend, color='b', label='model')
ax.plot(df_viz_copy.x, my_intercept + my_slope * df_viz_copy.x, color='crimson', label='truth')

ax.scatter(df.x, df.y, color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()
# %%
predictions = lm_fit.get_prediction(df_viz)
type(predictions)
# %%
lm_pred_summary = predictions.summary_frame()
type(lm_pred_summary)


# %%
lm_pred_summary
# %%
fig, ax = plt.subplots()


# prediction interval - the uncertainty on a single measurement (observation)
ax.fill_between(df_viz_copy.x,
                lm_pred_summary.obs_ci_lower, lm_pred_summary.obs_ci_upper,
                facecolor='orange', alpha=0.75, edgecolor='orange')

# include the true trend
ax.plot(df_viz_copy.x, my_intercept + my_slope * df_viz_copy.x, color='crimson')

# confidence interval - the unceratinty on the mean output
ax.fill_between(df_viz_copy.x,
                lm_pred_summary.mean_ci_lower, lm_pred_summary.mean_ci_upper,
                facecolor='grey', edgecolor='grey')

# output mean - the predictive trend - the best fit line
ax.plot(df_viz_copy.x, lm_pred_summary['mean'], color='k', linewidth=1.5)

# include the training data
ax.scatter(df.x, df.y, color='k', s=100)

# set the labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# show the plot
plt.show()
# %%
lm_pred_summary
# %%

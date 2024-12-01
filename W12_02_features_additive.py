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
fit_x1x2_add = smf.ols(formula='y ~ x1 + x2',data=df).fit()
fit_x1x2_add.params
# %%
fit_x1x2_add.bse
# %%
fit_x1x2_add.pvalues
# %%
fit_x1x2_add.pvalues<0.05
# %%
my_coefplot(fit_x1x2_add)
# %%
np.abs(fit_x1x2_add.params).sort_values(ascending=False)

# %%
df_viz_1 = pd.DataFrame({'x1': np.linspace(df.x1.min()-0.02, df.x1.max()+0.02, num=251)})
df_viz_1['x2'] = df.x2.mean()
df_viz_1
# %%
predictions_1 = fit_x1x2_add.get_prediction(df_viz_1)
pred_x1x2_add_summary = predictions_1.summary_frame()
pred_x1x2_add_summary
# %%
fit, ax = plt.subplots()
# prediction interval
ax.fill_between(df_viz_1.x1, 
                pred_x1x2_add_summary.obs_ci_lower, pred_x1x2_add_summary.obs_ci_upper,
                facecolor='orange', alpha=0.75, edgecolor='orange')

# confidence interval
ax.fill_between(df_viz_1.x1,
                pred_x1x2_add_summary.mean_ci_lower, pred_x1x2_add_summary.mean_ci_upper,
                facecolor='grey', edgecolor='grey')

# trend
ax.plot(df_viz_1.x1, pred_x1x2_add_summary['mean'], color='k', linewidth=1)

# set the labels
ax.set_xlabel('x1')
ax.set_ylabel('y')

# show the plot
plt.show()


# %%
input_grid = pd.DataFrame([(x1, x2) for x1 in np.linspace(df.x1.min(), df.x1.max(), num=101)
                                    for x2 in np.linspace(df.x2.min(), df.x2.max(), num=9)],
                          columns=['x1','x2'])
input_grid.info()
# %%
input_grid.nunique()
# %%
sns.relplot(data=input_grid, x='x1', y='x2', kind='scatter')
plt.show()
# %%
viz_grid = input_grid.copy() 
viz_grid['pred'] = fit_x1x2_add.predict(input_grid)
viz_grid
# %%
sns.relplot(data=viz_grid,
            x='x1', y='pred', kind='line',
            hue='x2', palette='coolwarm',
            estimator=None, units='x2')

plt.show()
# %%
fit_x1x2_add.params
# %%
import statsmodels.api as sm

# %%
fig, ax = plt.subplots()
sm.graphics.plot_fit(fit_x1x2_add, 'x1', ax=ax)
plt.show()
# %%
fig, ax = plt.subplots()
sm.graphics.plot_fit(fit_x1x2_add, 'x2', ax=ax)
plt.show()
# %%
fig = plt.figure(figsize=(16,8))
sm.graphics.plot_regress_exog(fit_x1x2_add, 'x1', fig=fig)
plt.show()
# %%
fig = plt.figure(figsize=(16,8))
sm.graphics.plot_regress_exog(fit_x1x2_add, 'x2', fig=fig)
plt.show()
# %%
fit_x1x2_add.params
# %%
b2
# %%
fit_x2.params
# %%
df.to_csv('linear_additive_example.csv', index=False)
# %%

# week11_review_lm_nonlinear
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

#%%
## Visualize Behavior

my_intercept = 0.25
my_slope = -2.25

df_viz = pd.DataFrame({'x':np.linspace(-3.14159, 3.14159, num=101)})
## Calculate the trend given the input
df_viz['trend'] = my_intercept + my_slope * np.sin(df_viz.x)

## Visualize the trend given the input
sns.set_style('whitegrid')
fig, ax = plt.subplots()
ax.plot(df_viz.x, df_viz.trend, color='crimson', linewidth=1.5)
ax.set_xlabel('x')
ax.set_ylabel('trend')
plt.show()
#%%
## If we generate the plot with sns
sns.relplot(data=df_viz, x='x', y='trend')
plt.show


# %%
fig, ax = plt.subplots()
ax.plot(np.sin(df_viz.x), df_viz.trend, color='crimson')

ax.set_xlabel('sin(x)')
ax.set_ylabel('trend')

plt.show()
# %%
my_sigma = 0.33

df_viz['obs_lwr_68'] = df_viz.trend - my_sigma
df_viz['obs_upr_68'] = df_viz.trend + my_sigma

df_viz['obs_lwr_95'] = df_viz.trend - 2 * my_sigma
df_viz['obs_upr_95'] = df_viz.trend + 2 * my_sigma

df_viz
# %%
fig, ax = plt.subplots()
# TRUE TREND
ax.plot(df_viz.x, df_viz.trend, color='crimson', linewidth=1.5)
# Variation around the trend - showing 2 intervals
# 2 sigma interval
ax.fill_between(df_viz.x, df_viz.obs_lwr_95, df_viz.obs_upr_95, facecolor = 'crimson', alpha=0.35)
# 1 sigma interval
ax.fill_between(df_viz.x, df_viz.obs_lwr_68, df_viz.obs_upr_68, facecolor = 'crimson', alpha = 0.35)

# set labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# show the plot
plt.show()


# %%

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
my_intercept = 0.25
my_slope = -1.5

# %%
df_viz = pd.DataFrame({'x':np.linspace(-3.5, 3.5, num=101)})
# %%
df_viz.head()
# %%
df_viz['trend'] = my_intercept + my_slope * df_viz.x
# %%
df_viz
# %%

## Seaborn chart
sns.set_style('whitegrid')
sns.relplot(data=df_viz, x='x', y='trend', kind='line')
plt.show()


# %%

## Matplotlib
fig, ax =plt.subplots()
ax.plot(df_viz.x, df_viz.trend)
ax.set_xlabel('x')
ax.set_ylabel('trend')
plt.show()
# %%
### Matplotlib change the color - black
fig, ax =plt.subplots()
ax.plot(df_viz.x, df_viz.trend, color='k')
ax.set_xlabel('x')
ax.set_ylabel('trend')
plt.show()
# %%
### Matplotlib change the color - red and linewidth = 3
fig, ax =plt.subplots()
ax.plot(df_viz.x, df_viz.trend, color='r', linewidth=3)
ax.set_xlabel('x')
ax.set_ylabel('trend')
plt.show()
# %%

### We will use the ±1σ and the ±2σ intervals around the mean. 
### These correspond to the 68% and the 95% uncertainty intervals.

my_sigma = 1.25

df_viz['obs_lwr_68'] = df_viz.trend - my_sigma
df_viz['obs_upr_68'] = df_viz.trend + my_sigma
# %%
df_viz['obs_lwr_95'] = df_viz.trend - 2 * my_sigma
df_viz['obs_upr_95'] = df_viz.trend + 2 * my_sigma
# %%
df_viz
# %%
fig, ax = plt.subplots()
# TRUE TREND
ax.plot(df_viz.x, df_viz.trend, color='crimson', linewidth=1.5)
# TRUE variation around the TREND - showing 2 intervals

# 2 simga interval
ax.fill_between(df_viz.x, df_viz.obs_lwr_95, df_viz.obs_upr_95, facecolor = 'crimson', alpha=0.35)
## COLOR: face color
## Transparency: alpha

# 1 sigma interval
ax.fill_between(df_viz.x, df_viz.obs_lwr_68, df_viz.obs_upr_68, facecolor='crimson', alpha=0.35)
# set labels
ax.set_xlabel('x')
ax.set_ylabel('y')
# show the plot
plt.show()
# %%
## Generate random output data
df = pd.DataFrame({'x':np.linspace(-3, 3, num=9)})
df['trend'] = my_intercept + my_slope * df.x

rg = np.random.default_rng(2100)
df['y'] = rg.normal(loc=df.trend, scale = my_sigma, size=df.shape[0])
df

# %%
## Visualize the randomly generated output around the TREND
fig, ax = plt.subplots()
# TRUE TREND
ax.plot(df_viz.x, df_viz.trend, color='crimson', linewidth=1.5)
# TRUE variation around the TREND - showing 2 intervals

# 2 simga interval
ax.fill_between(df_viz.x, df_viz.obs_lwr_95, df_viz.obs_upr_95, facecolor = 'crimson', alpha=0.35)
## COLOR: face color
## Transparency: alpha

# 1 sigma interval
ax.fill_between(df_viz.x, df_viz.obs_lwr_68, df_viz.obs_upr_68, facecolor='crimson', alpha=0.35)

# include the randomly generated small data set as a scatter plot
ax.scatter(df.x, df.y, color='k')

# set labels
ax.set_xlabel('x')
ax.set_ylabel('y')
# show the plot
plt.show()


# %%
df['true_intercept'] = my_intercept
df['true_slope'] = my_slope
df['true_sigma'] = my_sigma
df
# %%
# Save the dataframe to a CSV file
df.to_csv('week_11_linear_data.csv', index=False)
# %%
## When the folder is not we want.
import os
print("Current Working Directory:", os.getcwd())
#%%
## Let's start setting file directory
os.chdir(r"C:\Users\steve\OneDrive\桌面\PittMDS_DCC")
print("New Working Directory:", os.getcwd())
# %%

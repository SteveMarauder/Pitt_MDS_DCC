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
sns.set_style('whitegrid')
sns.relplot(data=df_viz, x='x', y='trend', kind='line')
plt.show()


# %%
fig, ax =plt.subplots()
ax.plot(df_viz.x, df_viz.trend)
plt.show()
# %%

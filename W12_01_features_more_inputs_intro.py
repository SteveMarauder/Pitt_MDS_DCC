#%%
### Introduction to working with more than 1 input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
#%%
### Additive
### Linear Relationships
def calc_trend_wrt_x1(x1, x2, b0, b1, b2):
    res_df = pd.DataFrame({'x1':x1})
    res_df['x2'] = x2
    res_df['trend'] = b0 + b1 * res_df.x1 + b2* res_df.x2
    return res_df

# To demonstrate the relationships, let's define the following COEFFICIENTS.
b0 = -0.25
b1 = 1.95
b2 = 0.2

# Let's now define 101 evenly or uniformly spaced values of `x1` between -3 and 3
x1_values = np.linspace(-3,3,num=101)
x1_values.size

x1_values.ndim
#%%
# Let's calculate the AVERAGE OUTPUT for a SINGLE value of `x1` at 0.
calc_trend_wrt_x1(x1_values, 0, b0, b2, b2)

#%%
calc_trend_wrt_x1(x1_values, 0, b0, b1, b2).x2.value_counts()
# %%

# Let's visualize the RELATIONSHIP between the AVERAGE OUTPUT and INPUT 1
sns.relplot(data = calc_trend_wrt_x1(x1_values, 0, b0, b1, b2),
            x='x1', y='trend', kind='line')

plt.show()


# %%
calc_trend_wrt_x1(x1_values, -2, b0, b1, b2)
# %%
sns.relplot(data=calc_trend_wrt_x1(x1_values, -2, b0, b1, b2 ),
            x = 'x1', y='trend', kind='line')
plt.show()
# %%
x2_values = np.linspace(-3,3,num=9)
x2_values

# %%
study_wrt_x1_list = [calc_trend_wrt_x1(x1_values, x2, b0, b1, b2) for x2 in x2_values]
len(study_wrt_x1_list)
# %%
study_wrt_x1_list[0]
# %%
study_wrt_x1_list[1]
# %%
study_wrt_x1_df = pd.concat(study_wrt_x1_list, ignore_index=True)
study_wrt_x1_df
# %%
study_wrt_x1_df.x2.value_counts()
# %%
## Visualize the TREND or AVERAGE OUTPUT with respect to x1 FOR EACH unique value of x2 AS A line chart
sns.relplot(data = study_wrt_x1_df,
            x='x1', y='trend', kind='line')
plt.show()
# %%
sns.relplot(data = study_wrt_x1_df,
            x='x1', y='trend', kind='scatter')
plt.show()
# %%
sns.relplot(data = study_wrt_x1_df,
            x='x1', y='trend', kind='line',
            estimator=None, units='x2')

plt.show()
# %%
sns.relplot(data = study_wrt_x1_df,
            x='x1', y='trend', kind='line',
            hue='x2', estimator=None, units='x2')
plt.show()
# %%
sns.relplot(data=study_wrt_x1_df,
            x='x1', y='trend', kind='line',
            hue='x2', palette='viridis', estimator=None, units='x2')
plt.show()
# %%
sns.relplot(data=study_wrt_x1_df,
            x='x1', y='trend', kind='line',
            hue='x2', palette='coolwarm', estimator=None, units='x2')
plt.show()
# %%

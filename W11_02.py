# week11_review_lm_nonlinear

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


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



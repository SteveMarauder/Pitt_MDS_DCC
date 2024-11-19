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

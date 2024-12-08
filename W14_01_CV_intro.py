## Introduction to Cross-Validation
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('week_13_binary_classification.csv')
df.info()
df.y.value_counts()
# %%
df.x5.value_counts()
# %%
df.nunique()
# %%
sns.lmplot(data=df, x='x1', y='y', logistic=True)
plt.show()
# %%
sns.lmplot(data = df, x='x2', y='y', logistic=True)
plt.show()
# %%
sns.lmplot(data = df, x='x2', y='y', logistic=True, ci=None)
plt.show()
# %%
sns.lmplot(data=df, x='x2', y='y', hue='x5', logistic=True, ci=None)
plt.show()
# %%
sns.lmplot(data=df, x='x1', y='y', hue='x5', logistic=True, ci=None)
plt.show()
# %%
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# %%
input_df = df.drop(columns=['y']).copy()
input_df.info()
# %%
type(input_df.to_numpy())

# %%
input_df.to_numpy().shape
# %%
df.y
# %%
df.y.to_numpy().shape
# %%
kf_a = KFold(n_splits=5, shuffle=True, random_state=101)
# %%
kf_a.get_n_splits()
# %%

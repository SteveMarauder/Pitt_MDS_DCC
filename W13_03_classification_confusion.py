#%%
# Measuring Classification Performance - Confusion Matrix
## import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
#%%
## Read data
df = pd.read_csv('week_12_intro_binary_classification.csv')
df
# %%
df.info()
# %%
fit_glm = smf.logit(formula='y ~ x', data=df).fit()
# %%
df_copy = df.copy()
df_copy['pred_probability'] = fit_glm.predict(df)

# %%
df_copy.head()
# %%
df_copy['pred_class'] = np.where(df_copy.pred_probability>0.5, 1, 0)
# %%
np.mean(df_copy.y == df_copy.pred_class)

# %%
df_copy.y.value_counts()
# %%
df_copy.pred_class.value_counts()
# %%
pd.crosstab(df_copy.y, df_copy.pred_class)

# %%
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class, margins=True),
            annot = True, annot_kws={'size':20}, fmt='3d')
plt.show()
# %%
df_copy.y.value_counts()
# %%
df_copy.pred_class.value_counts()
# %%
from sklearn.metrics import confusion_matrix
# %%
confusion_matrix(df_copy.y.to_numpy(), df_copy.pred_class.to_numpy())
# %%
confusion_matrix(df_copy.y.to_numpy(), df_copy.pred_class.to_numpy()).ravel()

#%%
TN, FP, FN, TP = confusion_matrix(df_copy.y.to_numpy(), df_copy.pred_class.to_numpy()).ravel()
# %%
TN
# %%
(TN + TP) / (TN+FP+FN+TP)
# %%
np.mean(df_copy.y == df_copy.pred_class)

# %%
FN + TP
# %%
df_copy.y.value_counts()
# %%
TP / (TP+FN)
# %%
TN / (TN+FP)
# %%
TN + FP
# %%
1 - (TN / (TN + FP))
# %%

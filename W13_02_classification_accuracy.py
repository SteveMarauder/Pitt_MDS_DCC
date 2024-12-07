#%%
# Measuring CLASSIFICATION PERFORMANCE - ACCURACY
## Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
#%%
## Read Data
df = pd.read_csv('week_12_intro_binary_classification.csv')
df.info()

# %%
## Fit the logistic regression model
fit_glm = smf.logit(formula='y ~ x', data=df).fit()

# %%
df_copy = df.copy()
df_copy['pred_probability'] = fit_glm.predict(df)
df_copy.info()
df_copy
# %%
df_copy.describe()
# %%
df_copy['pred_class'] = np.where(df_copy.pred_probability > 0.5, 1, 0)
df_copy.info()
# %%
df_copy.nunique()
# %%
df_copy.head()
# %%
df_copy['correct_class'] = df_copy.y == df_copy.pred_class
# %%
df_copy
# %%
df_copy.info()
# %%
df_copy.correct_class.sum()
# %%
df_copy.correct_class.value_counts()
# %%
df_copy.correct_class.value_counts(normalize=True)

# %%
df_copy.correct_class.mean()

# %%
df_copy
# %%
df_copy.y == df_copy.pred_class
# %%
np.mean(df_copy.y == df_copy.pred_class)
# %%

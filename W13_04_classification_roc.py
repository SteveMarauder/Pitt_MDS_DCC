#%%
## MEASRUING Classification Performance - ROC curve
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
fit_glm = smf.logit(formula='y ~ x', data=df).fit()
# %%
df_copy = df.copy()
df_copy['pred_probability'] = fit_glm.predict(df)
df
# %%
df_copy['pred_class'] = np.where(df_copy.pred_probability>0.5,1,0)
# %%
df_copy
# %%
np.mean(df_copy.y == df_copy.pred_class)
# %%
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class, margins=True),
            annot=True, annot_kws={'size':20}, fmt='3d')
plt.show()
# %%
from sklearn.metrics import confusion_matrix
TN, FP, FN, TP = confusion_matrix(df_copy.y.to_numpy(), df_copy.pred_class.to_numpy()).ravel()
# %%
Accuracy = (TN+TP)/(TN+FP+FN+TP)
Accuracy
# %%
Sensitivity = TP / (TP + FN)
Sensitivity
# %%
Specificity = TN/(TN+FP)
# %%
Specificity
# %%
FPR = 1 - Specificity
FPR
# %%
df_copy.head()
# %%
df_copy['pred_class_high_threshold'] = np.where(df_copy.pred_probability > 0.75, 1,0)
# %%
df_copy.head()
# %%
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class_high_threshold, margins=True),
           annot = True, annot_kws={'size':20}, fmt='3d', ax=ax)
plt.show()
# %%
fig, ax = plt.subplots(1,2,figsize=(18,7))
ax = ax.ravel()

sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class, margins=True),
           annot = True, annot_kws={'size':25}, fmt='3d', ax=ax[0])
ax[0].set_title('Threshold 0.5')

sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class_high_threshold, margins=True),
           annot = True, annot_kws={'size':25}, fmt='3d', ax=ax[1])
ax[1].set_title('Threshold 0.75')

plt.show()
# %%
np.mean(df_copy.y == df_copy.pred_class_high_threshold)

# %%
TN_higher, FP_higher, FN_higher, TP_higher = confusion_matrix(df_copy.y.to_numpy(), df_copy.pred_class_high_threshold.to_numpy()).ravel()

# %%
Accuracy_higher = (TN_higher + TP_higher) / (TN_higher + FP_higher + FN_higher + TP_higher)
Accuracy_higher
# %%
Sensitivity_higher = TP_higher / (TP_higher + FN_higher)
Sensitivity_higher

# %%
Specificity_higher = (TN_higher) / (TN_higher + FP_higher)
Specificity_higher
# %%
FPR_higher = 1 - Specificity_higher
FPR_higher
# %%
df_copy['pred_class_low_threshold'] = np.where(df_copy.pred_probability>0.25,1,0)
df_copy
# %%
fig, ax = plt.subplots()
sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class_low_threshold, margins=True),
            annot=True, annot_kws={'size':20}, fmt='3d',
            ax=ax)
plt.show()
# %%
fig, ax = plt.subplots(1,3,figsize=(18,7))
ax = ax.ravel()

sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class_low_threshold, margins=True),
           annot = True, annot_kws={'size':15}, fmt='3d', ax=ax[0])
ax[0].set_title('Threshold 0.25')

sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class, margins=True),
           annot = True, annot_kws={'size':15}, fmt='3d', ax=ax[1])
ax[1].set_title('Threshold 0.5')

sns.heatmap(pd.crosstab(df_copy.y, df_copy.pred_class_high_threshold, margins=True),
           annot = True, annot_kws={'size':15}, fmt='3d', ax=ax[2])
ax[2].set_title('Threshold 0.75')

plt.show()
# %%
TN_lower, FP_lower, FN_lower, TP_lower = confusion_matrix(df_copy.y.to_numpy(), df_copy.pred_class_low_threshold.to_numpy()).ravel()
TN_lower
#%%
FP_lower
#%%
FN_lower
#%%
TP_lower
# %%
Accuracy_lower = (TN_lower + TP_lower) / (TN_lower + FP_lower + FN_lower + TP_lower)
Accuracy_lower
# %%
Sensitivity_lower = (TP_lower) / (TP_lower + FN_lower)
Sensitivity_lower
# %%
Specificity_lower = (TN_lower)/(TN_lower + FP_lower)
Specificity_lower
# %%
FPR_lower = 1 - Specificity_lower
FPR_lower
# %%
my_roc = pd.DataFrame({'tpr':[Sensitivity_lower, Sensitivity, Sensitivity_higher],
                       'specificity': [Specificity_lower, Specificity, Specificity_higher],
                       'fpr': [FPR_lower, FPR, FPR_higher],
                       'threshold':[0.25,0.5,0.75]})
my_roc
# %%
sns.relplot(data = my_roc, x='fpr', y='tpr', s=300)
plt.show()
# %%
sns.relplot(data=my_roc, x='fpr', y='tpr', hue='threshold', s=300)
plt.show()
# %%
from sklearn.metrics import roc_curve

# %%
fpr_values, tpr_values, threshold_values = roc_curve(df_copy.y.to_numpy(), df_copy.pred_probability.to_numpy())
# %%
type(fpr_values)
# %%
fpr_values.ndim
# %%
fpr_values.shape
# %%
threshold_values[:11]
# %%
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(fpr_values, tpr_values, color='k')
ax.plot([0,1], [0,1], color='grey', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

plt.show()
# %%
from sklearn.metrics import roc_auc_score
roc_auc_score(df_copy.y.to_numpy(), df_copy.pred_probability.to_numpy())

# %%

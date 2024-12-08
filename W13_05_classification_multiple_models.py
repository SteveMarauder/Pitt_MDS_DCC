#%%
# Fitting and assessing the performance of multiple CLASSIFICATION models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

#%%
# We need 3 functions from SKLEARN to streamline calculating the classification performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
df = pd.read_csv('week_13_binary_classification.csv')
df.info()
# %%
df.nunique()
# %%
df.dtypes
# %%
df.y.value_counts()
# %%
df.y.value_counts(normalize=True)
# %%
df.y.mean()
# %%
1 - df.y.mean()
# %%
sns.catplot(data=df, x='x5', kind='count')
plt.show()
# %%
df.describe()
# %%
mod_aa = smf.logit(formula='y ~ 1', data=df).fit()
# %%
mod_aa.params
# %%
mod_bb = smf.logit(formula='y ~ x1+x2+x3+x4+x5', data=df).fit()

# %%
mod_bb.params
# %%
mod_bb.pvalues < 0.05
# %%
mod_cc = smf.logit(formula = 'y ~ x5 * (x1 + x2 + x3 + x4) + np.power(x1,2) + np.power(x2,2) + np.power(x3, 2) + np.power(x4, 2)', data=df).fit()
# %%
mod_cc.params
# %%
mod_cc.pvalues
# %%
mod_cc.pvalues<0.05
# %%
def fit_and_assess_logistic(mod_name, a_formula, train_data,threshold):
    a_mod = smf.logit(formula=a_formula, data=train_data).fit()
    train_copy = train_data.copy()
    train_copy['pred_probability'] = a_mod.predict(train_data)
    train_copy['pred_class'] = np.where(train_copy.pred_probability > threshold, 1, 0)
    TN, FP, FN, TP = confusion_matrix(train_copy.y.to_numpy(), train_copy.pred_class.to_numpy()).ravel()
    Accuracy = (TP+TN) / (TN+FP+FN+TP)
    Sensitivity = (TP) / (TP + FN)
    Specificity = (TN) / (TN + FP)
    FPR = 1 - Specificity
    ROC_AUC = roc_auc_score(train_copy.y.to_numpy(), train_copy.pred_probability.to_numpy())
    res_dict = {'model_name':mod_name,
                'model_formula': a_formula,
                'num_coefs': len(a_mod.params),
                'threshold': threshold,
                'Accuracy': Accuracy,
                'Sensitivity': Sensitivity,
                'Specificity': Specificity,
                'FPR': FPR,
                'ROC_AUC': ROC_AUC}
    return pd.DataFrame(res_dict, index=[0])


# %%
fit_and_assess_logistic(0, 'y~1', train_data=df, threshold=0.5)
# %%
formula_list = ['y ~ 1',
                'y ~ x5',
                'y ~ x1 + x2 + x3 + x4',
                'y ~ x1 + x2 + x3 + x4 + x5',
                'y ~ x5 * (x1 + x2 + x3 + x4)',
                'y ~ (x1 + x2 + x3 + x4) ** 2',
                'y ~ x1 + x2 + x3 + x4 + np.power(x1,2)+np.power(x2,2)+np.power(x3,2)+np.power(x4,2)',
                'y ~ x5 + x1 + x2 + x3 + x4 + np.power(x1,2)+np.power(x2,2)+np.power(x3,2)+np.power(x4,2)',
                'y ~ x5 * (x1 + x2 + x3 + x4 + np.power(x1,2)+np.power(x2,2)+np.power(x3,2)+np.power(x4,2))',
                'y ~ x5 + ((x1 + x2 + x3 + x4)**2 + np.power(x1,2)+np.power(x2,2)+np.power(x3,2)+np.power(x4,2))',
                'y ~ x5 * ((x1 + x2 + x3 + x4)**2 + np.power(x1,2)+np.power(x2,2)+np.power(x3,2)+np.power(x4,2))',
                'y ~ x5 + (x1 + x2 + x3 + x4)**3',
                'y ~ x5 + (x1 + x2 + x3 + x4)**4',
                'y ~ (x1 + x2 + x3 + x4 + x5)**3',
                'y ~ (x1 + x2 + x3 + x4 + x5)**4',
                'y ~ x5 * ((x1 + x2 + x3 + x4)**3 + np.power(x1,2) + np.power(x2, 2) + np.power(x3,2) + np.power(x4, 2) + np.power(x1, 3) + np.power(x2, 3) + np.power(x3, 3) + np.power(x4, 3))',
                'y ~ x5 * ((x1 + x2 + x3 + x4)**3 + np.power(x1,2) + np.power(x2, 2) + np.power(x3,2) + np.power(x4, 2) + np.power(x1, 3) + np.power(x2, 3) + np.power(x3, 3) + np.power(x4, 3)+ np.power(x1, 4) + np.power(x2, 4) + np.power(x3, 4) + np.power(x4, 4))']

# %%
formula_list[0]
# %%
formula_list[2]
# %%
formula_list[3]
# %%
results_list = []

for m in range(len(formula_list)):
    results_list.append(fit_and_assess_logistic(m, formula_list[m], train_data=df, threshold=0.5))

# %%
len(results_list)
# %%
len(formula_list)
# %%
results_list[0]
# %%
results_list[1]
# %%
results_df = pd.concat(results_list, ignore_index = True)
results_df
# %%
results_df.sort_values(by=['Accuracy'], ascending=False)
# %%
results_df.sort_values(by=['ROC_AUC'], ascending=False)
# %%
sns.relplot(data=results_df, x='num_coefs', y='Accuracy', kind='scatter')
plt.show()
# %%
sns.relplot(data=results_df, x='num_coefs', y='ROC_AUC', kind='scatter')
plt.show()
# %%
def fit_logistic_make_roc(mod_name, a_formula, train_data):
    a_mod = smf.logit(formula=a_formula, data=train_data).fit()
    train_copy = train_data.copy()
    train_copy['pred_probability'] = a_mod.predict(train_data)
    fpr, tpr, threshold = roc_curve(train_data.y.to_numpy(), train_copy.pred_probability.to_numpy())
    res_df = pd.DataFrame({'tpr':tpr,
                           'fpr':fpr,
                           'threshold': threshold                              
                           })
    res_df['model_name'] = mod_name
    res_df['model_formula'] = a_formula
    return res_df

# %%
roc_list = []
for m in range(len(formula_list)):
    roc_list.append(fit_logistic_make_roc(m, formula_list[m], train_data=df))


# %%
roc_df = pd.concat(roc_list, ignore_index=True)
roc_df.info()
# %%
roc_df['model_name'] = roc_df.model_name.astype('category')
roc_df.info()
# %%
sns.relplot(data=roc_df, x='fpr', y='tpr', hue='model_name',
            kind='line', estimator=None, units='model_name')
plt.show()
# %%
sns.relplot(data=roc_df, x='fpr', y='tpr', hue='model_name',
            kind='line', estimator=None, units='model_name',
            col='model_name', col_wrap=5)
plt.show()
# %%
df
# %%
input_grid = pd.DataFrame([(x1,x2,x3,x4,x5) for x1 in np.linspace(df.x1.min(), df.x1.max(), num=101)
                                            for x2 in np.linspace(df.x2.min(), df.x2.max(), num=9)
                                            for x3 in [df.x3.mean()]
                                            for x4 in [df.x4.mean()]
                                            for x5 in df.x5.unique()],
                            columns=['x1','x2','x3','x4','x5'])
# %%
input_grid
# %%
input_grid.shape
# %%
input_grid.nunique()
# %%
input_grid.x5.value_counts()

# %%
sns.relplot(data=input_grid, x='x1', y='x2', col='x5')
plt.show()
# %%
mod_bb.params
# %%
dfviz = input_grid.copy()
dfviz['pred_probability_bb'] = mod_bb.predict(input_grid)
dfviz
# %%
sns.relplot(data=dfviz, x='x1', y='pred_probability_bb', hue='x5', col='x2',
            kind='line', estimator=None, units='x5',
            col_wrap=3)
           
plt.show()
# %%
mod_bb.params
# %%
mod_bb.pvalues<0.05
# %%
formula_list[len(formula_list)-1]
# %%
mod_complex=smf.logit(formula=formula_list[len(formula_list)-1], data=df).fit()
# %%
mod_complex.params
# %%
mod_complex.params.to_numpy()
# %%
dfviz['pred_probability_complex'] = mod_complex.predict(input_grid)
sns.relplot(data=dfviz, x='x1', y='pred_probability_complex', hue='x5', col='x2',
            kind='line', estimator=None, units='x5',
            col_wrap=3)
plt.show()
# %%

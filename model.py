import yaml
import dill
import functools

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer, scale
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier

from lib import *

# set random seed
np.random.seed(1)

# import our data
admit = pd.read_csv('data/diabetic_data.csv', na_filter=True, na_values=['?', 'None'])

# encode our y variable
admit['readmitted'] = admit.readmitted.apply(lambda x: 1 if x == '<30' else 0)

# train / test split
xdata = admit.drop(labels=['readmitted'], axis=1)
ydata = admit.readmitted
xdata_train, xdata_test, ydata_train, ydata_test = train_test_split(xdata, ydata, test_size=0.2, random_state=1)

# set up pipelines
base = [
    ('xv', FunctionTransformer(lambda X: X[xvar].copy(), validate=False)),
    ('md', FunctionTransformer(med_features, validate=False)),
    ('df', FunctionTransformer(directional_features, validate=False)),
    ('dx', FunctionTransformer(diag_features, validate=False))
]

fe1 = [
    ('oh', OneHotEncoder(columns=cat_var, iid=True, column_params={'min_support': 30}, pandas_out=True)),
    ('hc', HCCEncoder(columns=hcc_cat_var, iid=True, column_params={'add_noise': False})),
    ('im', Imputer(missing_values='NaN', strategy='median')),
    ('ss', StandardScaler())
]

fe2 = [
    ('dp', FunctionTransformer(lambda X: X.drop(labels=hcc_cat_var, axis=1), validate=False)),
    ('oh', OneHotEncoder(columns=cat_var, iid=True, column_params={'min_support': 30}))
]

model_stack = [
    base + fe1 + [('lr', LogisticRegression(random_state=1, class_weight="balanced"))],
    base + fe2 + [('rf', RandomForestClassifier(random_state=1, class_weight="balanced"))],
    base + fe2 + [('gb', XGBClassifier(seed=1, scale_pos_weight=(1/np.mean(ydata_train)-1)))]
]

model_stack = [(m[-1][0], Pipeline(steps=m)) for m in model_stack]

# hyperparameter tuning for each model individually
tuning_constants = {'scoring': 'roc_auc', 'cv': 3, 'verbose': 1, 'refit': False}
grid_search_tuning_arg = tuning_constants.copy()
rand_search_tuning_arg = dict(tuning_constants, **{'random_state': 1, 'n_iter': 20})
tuning_types = {'lr': GridSearchCV, 'rf': RandomizedSearchCV, 'gb': RandomizedSearchCV}

def make_tuner(cls, pipeline, params):
    kwarg = grid_search_tuning_arg if cls is GridSearchCV else rand_search_tuning_arg
    return cls(pipeline, params, **kwarg)

param_grid = {
    'lr': {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1]
    },
    'rf': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, None],
        'min_samples_leaf': [5, 10, 20],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    },
    'gb': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.02, 0.05, 0.1],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.75, 1],
        'colsample_bytree': [0.5, 0.75, 1]
    }
}

try:
    with open('model_param.yaml', 'r') as f:
        param_optimal = yaml.load(f)
except IOError:
    param_optimal = {}

    for m in model_stack:
        # create tuner
        model_name, pipeline = m
        param_grid_model = add_dict_prefix(param_grid[model_name], model_name)
        tuner = make_tuner(tuning_types[model_name], pipeline, param_grid_model)

        # use tuner to determine optimal params
        tuner.fit(xdata_train, ydata_train)
        print('Best %s params: %s' % (model_name, str(tuner.best_params_)))
        print('Best %s params score: %s' % (model_name, str(tuner.best_score_)))

        # save best params
        param_optimal.update(**add_dict_prefix(tuner.best_params_, model_name))

    with open('model_param.yaml', 'w') as f:
        yaml.dump(param_optimal, f)

# build model stack
stack = StackingClassifier(classifiers=model_stack)
stack.set_params(**param_optimal)
stack.fit(xdata_train, ydata_train)

# make predictions for our test set
ydata_test_pred = stack.predict_proba(xdata_test)[:,1]

# determine cutoff balancing precision/recall
precision, recall, threshold = precision_recall_curve(ydata_test, ydata_test_pred)
pos_threshold = np.min(threshold[precision[1:] > recall[:-1]])
print('Positive threshold: %s' % str(pos_threshold))
print('Confusion matrix:')
print(confusion_matrix(ydata_test, (ydata_test_pred >= pos_threshold).astype(int)))
print('Stack AUC: %s' % roc_auc_score(ydata_test, ydata_test_pred))

# ensemble versus individual models
pred = []
for m in stack.named_steps['stack'].transformer_list:
    model_name, model = m
    pred_i = model.transform(xdata_test)
    pred.append(pred_i)
    print('%s AUC: %s' % (model_name.upper(), roc_auc_score(ydata_test, pred_i)))

avg_pred = np.average(pred, axis = 0)
print('Avg AUC: %s' % roc_auc_score(ydata_test, avg_pred))

# importance scores
lr_model = stack.named_steps['stack'].transformer_list[0][1]
rf_model = stack.named_steps['stack'].transformer_list[1][1]
xgb_model = stack.named_steps['stack'].transformer_list[2][1]

lr_features = lr_model.named_steps['hc'].get_feature_names()
tree_features = rf_model.named_steps['oh'].get_feature_names()

lr_coef = lr_model.named_steps['lr'].coef_[0]
rf_coef = rf_model.named_steps['rf'].feature_importances_
xgb_coef = xgb_model.named_steps['gb'].feature_importances_

i1 = pd.DataFrame(data={'feature': lr_features, 'lr_coef': lr_coef})
i2 = pd.DataFrame(data={'feature': tree_features, 'rf_imp': rf_coef})
i3 = pd.DataFrame(data={'feature': tree_features, 'xgb_imp': xgb_coef})

merge = functools.partial(pd.merge, on=['feature'], how="outer")
importance = functools.reduce(merge, [i1, i2, i3])
importance = importance.fillna(0)
importance['overall'] = (scale(abs(importance.lr_coef)) + scale(importance.rf_imp) + scale(importance.xgb_imp))/3
importance.sort_values(by=['overall'], ascending=False, inplace=True)
print(importance)

# pickle
with open('model.pkl', 'wb') as f:
    dill.dump(stack, f)

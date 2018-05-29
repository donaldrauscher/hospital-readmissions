import dill
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from lib import *

# load model
with open('model.pkl', 'rb') as f:
    model = dill.load(f)

# import data
admit = pd.read_csv('data/diabetic_data.csv', na_filter=True, na_values=['?', 'None'])

# generate predictions and score
ydata = admit.readmitted.apply(lambda x: 1 if x == '<30' else 0)
ydata_pred = model.predict_proba(admit)[:,1]
print(roc_auc_score(ydata, ydata_pred))


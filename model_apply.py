import argparse
import os
import dill
import pandas as pd
import numpy as np
from lib import *

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

# load model
basedir = os.getenv("MOUNT", "/usr/share/model")
with open(os.path.join(basedir, 'model.pkl'), 'rb') as f:
    model = dill.load(f)

# load data
df = pd.read_csv(os.path.join(basedir, args.input), na_filter=True, na_values=['?', 'None'])

# generate predictions and score
pred = model.predict_proba(df)
pred.to_csv(os.path.join(basedir, 'output/predictions.csv', index=False))



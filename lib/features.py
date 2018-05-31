import pandas as pd
import numpy as np

from .util import get_first
from .variables import med_var


# encode medication variables
def med_features(df):
    for m in med_var:
        df['has_' + m] = df[m].apply(lambda x: 0 if x == 'No' else 1)
        df['dir_' + m] = df[m].apply(lambda x: -1 if x == 'Down' else 1 if x == 'Up' else 0)

    df.drop(labels=med_var, axis=1, inplace=True)
    df['diabetesMed'] = df.diabetesMed.apply(lambda x: 1 if x == 'Yes' else 0)
    df['change'] = df.change.apply(lambda x: 1 if x == 'Ch' else 0)
    return df


# encode directional variables
def directional_encode(df, col, val):
    has_null = np.sum(df[col].isnull())
    if has_null:
        df['has_' + col] = df[col].isnull().astype(int)
        df.loc[df[col].isnull(), [col]] = 0
    for i,v in enumerate(val):
        df.loc[df[col].astype(str) == v, [col]] = i
    return df


def make_range(start, stop, increment, pattern="[%s-%s)"):
    r = np.arange(start, stop + 1, increment)
    r1, r2 = r[:-1], r[1:]
    r = [pattern % (i[0], i[1]) for i in zip(r1, r2)]
    r = r + ['>%s' % (stop)]
    return r


def directional_features(df):
    df = directional_encode(df, 'A1Cresult', ['Norm','>7','>8'])
    df = directional_encode(df, 'max_glu_serum', ['Norm','>200','>300'])
    df = directional_encode(df, 'weight', make_range(0, 200, 25))
    df = directional_encode(df, 'age', make_range(0, 100, 10))
    return df


# combine diagnosis codes into array; remove missing
def diag_features(df):
    df = df.reset_index(drop=True)
    diag = df[['diag_1', 'diag_2', 'diag_3']].values
    diag = [x[~pd.isnull(x)] for x in diag]
    df['diag'] = pd.Series(diag)
    df['diag_first'] = pd.Series([get_first(x) for x in diag])
    df.drop(labels=['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)
    return df
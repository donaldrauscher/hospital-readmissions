import pandas as pd
import numpy as np

# flatten array of items, some of which can be arrays
def flatten(y):
    y = [i if isinstance(i, (np.ndarray, list, tuple)) else [i] for i in y]
    cuts = list(np.cumsum([len(i) for i in y])[:-1])
    flat = [item for sublist in y for item in sublist]
    return flat, cuts if len(flat) > (len(cuts) + 1) else None

# unflattens array
def unflatten(y, cuts):
    return np.split(y, cuts) if cuts else y

# add prefix to keys in a dictionary
def add_dict_prefix(x, px):
    return {'%s__%s' % (px, k) : v for k,v in x.items()}

# gets the first element in array if exists
def get_first(x):
    try:
        return x[0]
    except IndexError:
        pass

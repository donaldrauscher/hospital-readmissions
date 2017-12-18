import pandas as pd
import numpy as np

from scipy.optimize import minimize
from scipy.special import beta

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, column_or_1d
from sklearn.utils.multiclass import type_of_target

# beta binomial density function
@np.vectorize
def dbetabinom(a, b, k, n):
    n2 = np.clip(n, 0, 100)
    k2 = round(k * n2 / n)
    return beta(k2 + a, n2 - k2 + b) / beta(a, b)

# beta binomial log loss
def betabinom_ll(par, arg):
    return np.sum(-np.log(dbetabinom(par[0], par[1], arg[0], arg[1])))

# default params for MLE
mle_param_defaults = dict(method = "L-BFGS-B", x0 = [1,1], bounds = [(0.5, 500), (0.5, 500)])

# encodes single high cardinality categorical variable
class SingleHCCEncoder(BaseEstimator):

    def __init__(self, add_noise = True, noise_sd = 0.05, mle_params = mle_param_defaults):
        self.add_noise = add_noise
        self.noise_sd = noise_sd
        self.mle_params = mle_params
        self.a, self.b = None, None
        self.df, self.df_dict = None, None

    # calibrate a and b of beta distribution
    def fit_beta(self):
        check_is_fitted(self, 'df')
        k, n = self.df.k, self.df.n
        mle = minimize(fun = betabinom_ll, args = [k, n], **self.mle_params)
        self.a, self.b = mle.x

    # descriptive stats for each level
    def fit_df(self, x, y):
        df = pd.DataFrame(data = dict(x = x, y = y))
        df = df.groupby(['x']).agg(['sum', 'count', 'mean'])
        df.columns = ['k', 'n', 'p']
        self.df = df
        self.df_dict = df.to_dict(orient = "index")

    @np.vectorize
    def transform_one_loo(self, x, y):
        xval = self.df_dict.get(x, dict(k = 0, n = 0))
        return (xval['k'] + self.a - y) * 1.0 / (xval['n'] + self.a + self.b - 1)

    @np.vectorize
    def transform_one(self, x):
        xval = self.df_dict.get(x, dict(k = 0, n = 0))
        return (xval['k'] + self.a) * 1.0 / (xval['n'] + self.a + self.b)

    def fit(self, x, y):
        assert(type_of_target(y) == "binary")
        x = column_or_1d(x)
        y = column_or_1d(y)
        self.fit_df(x, y)
        self.fit_beta()
        return self

    def fit_transform(self, x, y):
        self.fit(x, y)
        if self.add_noise:
            noise = self.noise_sd * np.random.randn(len(x)) + 1
        else:
            noise = np.repeat(1, len(x))
        return self.transform_one_loo(self, x, y) * noise

    def transform(self, x):
        check_is_fitted(self, 'df_dict')
        x = column_or_1d(x)
        return self.transform_one(self, x)

# encodes multiple high cardinality categorical variables
class HCCEncoder(BaseEstimator):

    def __init__(self, columns, hcc_encode_params = {}, seed = 1):
        self.columns = columns
        self.hcc_encode_params = hcc_encode_params
        self.seed = seed
        self.labellers = {}

    def fit(self, df, y):
        for c in self.columns:
            hcc_encode_params = self.hcc_encode_params.get(c, {})
            labeller = SingleHCCEncoder(**hcc_encode_params)
            labeller.fit(df[c], y)
            self.labellers[c] = labeller
        return self

    def fit_transform(self, df, y):
        np.random.seed(self.seed)
        df = df.copy()
        for c in self.columns:
            hcc_encode_params = self.hcc_encode_params.get(c, {})
            labeller = SingleHCCEncoder(**hcc_encode_params)
            df[c] = labeller.fit_transform(df[c], y)
            self.labellers[c] = labeller
        return df

    def transform(self, df):
        df = df.copy()
        for c in self.columns:
            df[c] = self.labellers[c].transform(df[c])
        return df

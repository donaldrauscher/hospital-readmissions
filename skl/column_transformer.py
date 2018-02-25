import pandas as pd
import numpy as np

from scipy import sparse

from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _transform_one, _fit_transform_one
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed


# only train with select columns
def select_x(X, name):
    return X[name] if name != 'other_col' else X


# class which applies column operations to multiple columns
class ColumnTransformer(FeatureUnion):

    def __init__(self, columns, transformer, transformer_params = {}, n_jobs = 1, multicol = False):

        # create transformer list
        transformer_list = [('other_col', ColumnFeedThrough(columns))]
        for c in columns:
            transformer_params = transformer_params.get(c, {})
            transformer_c = transformer(**transformer_params)
            transformer_list.append((c, transformer_c))

        # set
        self.columns = columns
        self.transformer = transformer
        self.transformer_params = transformer_params
        self.transformer_list = transformer_list
        self.transformer_weights = None
        self.n_jobs = n_jobs
        self.multicol = multicol

        # validate
        self._validate_transformers()

    def get_feature_names(self):
        if self.multicol:
            return super(ColumnGenerator, self).get_feature_names()
        else:
            return self.transformer_list[0][-1].col + self.columns

    def fit(self, X, y=None):
        self._validate_transformers()
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, select_x(X, name), y)
            for name, trans, _ in self._iter())

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, select_x(X, name), y, **fit_params)
            for name, trans, weight in self._iter())

        # All transformers are None
        if not result:
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.column_stack(Xs)
        return Xs

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, select_x(X, name))
            for name, trans, weight in self._iter())

        # All transformers are None
        if not Xs:
            return np.zeros((X.shape[0], 0))

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.column_stack(Xs)
        return Xs


# selects columns
class ColumnFeedThrough(BaseEstimator, TransformerMixin):

    def __init__(self, drop_col=[]):
        self.drop_col = drop_col
        self.col = []

    def fit(self, X, y = None):
        self.col = [x for x in list(X.columns) if x not in self.drop_col]
        return self

    def transform(self, X):
        return X[self.col]

    def get_feature_names(self):
        return self.col
import pandas as pd
import numpy as np

import abc

from scipy import sparse

from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _transform_one, _fit_transform_one
from sklearn.externals.joblib import Parallel, delayed


# class which applies column operations to multiple columns
class ColumnTransformer(FeatureUnion):

    def __init__(self, columns, column_params={}, iid=False, n_jobs=1, pandas_out=False):

        # create transformer list
        transformer_list = []
        for c in columns:
            column_params_c = column_params if iid else column_params.get(c, {})
            transformer_c = self.transformer(**column_params_c)
            transformer_list.append((c, transformer_c))

        # set
        self.columns = columns
        self.column_params = column_params
        self.non_columns = None
        self.iid = iid
        self.transformer_list = transformer_list
        self.transformer_weights = None
        self.n_jobs = n_jobs
        self.pandas_out = pandas_out

        # validate
        self._validate_transformers()

    @abc.abstractmethod
    def transformer(self):
        pass

    @abc.abstractmethod
    def multi_col(self):
        pass

    def get_feature_names(self):
        if self.multi_col:
            feature_names = []
            for name, trans in self.transformer_list:
                if not hasattr(trans, 'get_feature_names'):
                    raise AttributeError("Transformer %s (type %s) does not "
                                         "provide get_feature_names."
                                         % (str(name), type(trans).__name__))
                feature_names.extend([name + "__" + f for f in trans.get_feature_names()])
            feature_names.extend(self.non_columns)
        else:
            feature_names = self.columns + self.non_columns
        return feature_names

    def output(self, X):
        if self.pandas_out:
            return pd.DataFrame(data=X, columns=self.get_feature_names())
        return X

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.non_columns = [c for c in X.columns if c not in self.columns]

        self._validate_transformers()
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X[name], y)
            for name, trans, _ in self._iter())

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        assert isinstance(X, pd.DataFrame)
        self.non_columns = [c for c in X.columns if c not in self.columns]

        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X[name], y, **fit_params)
            for name, trans, weight in self._iter())

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xs = tuple(list(Xs) + [X[self.non_columns]])

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.column_stack(Xs)

        return self.output(Xs)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X[name])
            for name, trans, weight in self._iter())

        Xs = tuple(list(Xs) + [X[self.non_columns]])

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.column_stack(Xs)

        return self.output(Xs)

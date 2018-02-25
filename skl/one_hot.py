import pandas as pd
import numpy as np

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d

from util import flatten, unflatten

from skl.column_transformer import ColumnTransformer


# labeler which handles missing labels
class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, top_n = np.inf, min_support = 30):
        self.top_n = top_n
        self.min_support = min_support

    # include 'other' category
    @property
    def classes(self):
        return [str(c) for c in self.classes_.tolist()] + ['other']

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        y2, _ = flatten(y)

        classes = pd.Series(y2).value_counts()
        classes = classes[classes >= self.min_support]
        if not np.isinf(self.top_n):
            classes = classes.head(self.top_n)
        classes = np.sort(classes.index)

        self.classes_ = classes

        return self

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)
        y2, y2_cuts = flatten(y)

        new_label = ~np.in1d(y2, self.classes_)
        labels = np.searchsorted(self.classes_, y2)
        labels[new_label] = len(self.classes_)

        return unflatten(labels, y2_cuts)


# one-hot encoder for a single column
class OneHotEncoderBase(BaseEstimator, TransformerMixin):

    def __init__(self, **label_encoder_params):
        self.label_encoder = LabelEncoder(**label_encoder_params)

    def fit(self, X, y = None):
        self.label_encoder.fit(X)
        return self

    def transform(self, X):
        n_row, n_col = len(X), len(self.label_encoder.classes)
        X2 = self.label_encoder.transform(X)

        df_col, _ = flatten(X2)
        if len(df_col) > len(X):
            df_row = np.repeat(np.arange(len(X)), [len(i) for i in X])
        else:
            df_row = np.arange(len(X))
        df_val = np.ones(len(df_col))

        df = sparse.coo_matrix((df_val, (df_row, df_col)), shape=(n_row, n_col)).tocsr()
        df = df.tocsr().toarray()
        return df[:,:-1]

    def get_feature_names(self):
        return self.label_encoder.classes[:-1]


# one-hot encoder
class OneHotEncoder(ColumnTransformer):

    def __init__(self, columns, transformer_params = {}, n_jobs = 1, pandas_out = True):
        super(OneHotEncoder, self).__init__(columns = columns,
                                            transformer = OneHotEncoderBase,
                                            transformer_params = transformer_params,
                                            n_jobs = n_jobs,
                                            multi_col = True,
                                            pandas_out = pandas_out)
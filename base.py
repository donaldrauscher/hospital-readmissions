import pandas as pd
import numpy as np

from collections import OrderedDict

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d

from util import flatten, unflatten

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


# does one-hot encoding for multiple categoricals
class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns, label_encode_params = {}):
        self.columns = columns
        self.label_encode_params = label_encode_params
        self.df_columns = None
        self.labellers = OrderedDict()
        self.feature_indices = None

    def fit(self, df, y = None):
        # split out categoricals
        df1, df2 = df.drop(labels = self.columns, axis = 1).copy(), df[self.columns].copy()

        # label encode categoricals
        for c in self.columns:
            label_encode_params = self.label_encode_params.get(c, {})
            labeller = LabelEncoder(**label_encode_params)
            df2[c] = labeller.fit_transform(df2[c])
            self.labellers[c] = labeller

        # set number of values / feature indices for each categorical
        n_values = [len(self.labellers[c].classes) for c in self.columns]
        self.feature_indices = np.hstack([[0], np.cumsum(n_values)])

        # set columns
        non_cat_col = list(df1.columns)
        new_cat_col, _ = flatten([['is_%s_%s' % (k, i) for i in v.classes] for k,v in self.labellers.items()])
        new_cat_col = [j for i,j in enumerate(new_cat_col) if i not in self.feature_indices[1:]-1]
        self.df_columns = non_cat_col + new_cat_col

        return self

    def transform(self, df):
        # split out categoricals
        df1, df2 = df.drop(labels = self.columns, axis = 1).copy(), df[self.columns].copy()

        # apply transformations to each column
        feature_indices = dict(zip(self.columns, self.feature_indices[:-1]))
        for c in self.columns:
            df2[c] = self.labellers[c].transform(df2[c]) + feature_indices[c]

        # create matrix with one hots
        n_row, n_col = len(df2), self.feature_indices[-1]

        df2_flat = [flatten(x)[0] for x in df2.values]

        df2_col, _ = flatten(df2_flat)
        df2_row = np.repeat(np.arange(len(df2_flat)), [len(x) for x in df2_flat])
        df2_val = np.ones(len(df2_col))

        df2 = sparse.coo_matrix((df2_val, (df2_row, df2_col)),
                                shape=(n_row, n_col)).tocsr().toarray()

        # convert back into df and name columns
        col_names, _ = flatten([['is_%s_%s' % (k, i) for i in v.classes] for k,v in self.labellers.items()])
        df2 = pd.DataFrame(df2, columns = col_names)
        redundant = [col_names[i-1] for i in self.feature_indices[1:]]
        df2 = df2.drop(labels = redundant, axis = 1)

        # combine
        df1.reset_index(inplace = True, drop = True)
        df2.reset_index(inplace = True, drop = True)
        df = pd.concat([df1, df2], axis = 1)
        assert(list(df.columns) == self.df_columns)
        return df

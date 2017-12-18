import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

from scipy.special import logit

# method for linking `predict_proba` to `transform`
def chop_col0(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)[:,1:]
    return wrapper

def add_transform(classifiers):
    for key, classifier in classifiers:
        if isinstance(classifier, Pipeline):
            classifier = classifier.steps[-1][-1]
        classifier.transform = chop_col0(classifier.predict_proba)
        classifier.__class__.transform = chop_col0(classifier.__class__.predict_proba)
        # NOTE: need to add to class so `clone` in `cross_val_predict` works

# default function applies logit to probabilies and applies logistic regression
def default_meta_classifier():
    return Pipeline([
        ('logit', FunctionTransformer(logit)),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression())
    ])

# stacking classifier
class StackingClassifier(Pipeline):

    def __init__(self, classifiers, meta_classifier = None, cv = 3):
        add_transform(classifiers)
        self.classifiers = FeatureUnion(classifiers)
        self.meta_classifier = meta_classifier if meta_classifier else default_meta_classifier()
        self.cv = cv
        self.steps = [('stack', self.classifiers), ('meta', self.meta_classifier)]
        self.memory = None

    def fit(self, X, y):
        meta_features = cross_val_predict(self.classifiers, X, y, cv = self.cv, method = "transform")
        self.meta_classifier.fit(meta_features, y)
        self.classifiers.fit(X, y)

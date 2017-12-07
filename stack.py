from mlxtend.classifier import StackingClassifier as BaseStackingClassifier
from mlxtend.externals.name_estimators import _name_estimators

class StackingClassifier(BaseStackingClassifier):

    def __init__(self, classifiers, meta_classifier,
                 use_probas=False, average_probas=False, verbose=0,
                 use_features_in_secondary=False):

        if type(classifiers[0]) is tuple:
            self.classifiers = [c[1] for c in classifiers]
            self.named_classifiers = dict(classifiers)
        else:
            self.classifiers = classifiers
            self.named_classifiers = {k: v for k,v in _name_estimators(classifiers)}

        if type(meta_classifier) is tuple:
            self.meta_classifier = meta_classifier[1]
            self.named_meta_classifier = dict([meta_classifier])
        else:
            self.meta_classifier = meta_classifier
            self.named_meta_classifier = {'meta-%s' % k: v for k,v in _name_estimators([meta_classifier])}

        self.use_probas = use_probas
        self.average_probas = average_probas
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary

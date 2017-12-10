import numpy as np

from sklearn.ensemble import VotingClassifier as BaseVotingClassifier
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.metrics.scorer import check_scoring, _PredictScorer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict

# https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
# https://gist.github.com/thebertster/0362ae4c8197f0d6bee10d484b509912
def stars_and_bars(bins, stars, allowEmpty = True):
    # validate inputes
    if bins < 1 or stars < 1:
        raise ValueError("Number of objects (stars) and bins must both be greater than or equal to 1.")
    if not allowEmpty and stars < bins:
        raise ValueError("Number of objects (stars) must be greater than or equal to the number of bins.")

    # if there is only one bin, there is only one arrangement!
    if bins == 1:
        yield stars,
        return

    # if empty bins are not allowed, distribute (star-bins) stars and add an extra star to each bin when yielding.
    if not allowEmpty:
        if stars == bins:
            # If same number of stars and bins, then there is only one arrangement!
            yield tuple([1] * bins)
            return
        else:
            stars -= bins

    # 'bars' holds the queue or stack of positions of the bars in the stars and bars arrangement
    # (including a bar at the beginning and end) and the level of iteration that this stack item has reached.
    # Initial stack holds a single arrangement ||...||*****...****| with an iteration level of 1.
    bars = [([0]*bins + [stars], 1)]

    # iterate through the current queue of arrangements until no more are left (all arrangements have been yielded)
    while len(bars) > 0:
        newBars = []

        for b in bars:
            # iterate through inner arrangements of b, yielding each arrangement and queuing each
            # arrangement for further iteration except the very first
            for x in range(b[0][-2], stars+1):
                newBar = b[0][1:bins] + [x, stars]
                if b[1] < bins-1 and x > 0:
                    newBars.append((newBar, b[1]+1))

                # translate the stars and bars into a tuple
                yield tuple(newBar[y] - newBar[y-1] + (0 if allowEmpty else 1) for y in range(1, bins+1))

        bars = newBars

# passes pre-computed outputs to scorer
class ScoreEstPassThru(object):

    def __init__(self, y_label_encoder):
        self.y_label_encoder = y_label_encoder

    def predict(self, X):
        check_is_fitted(self.y_label_encoder)
        maj = np.argmax(X, axis = 1)
        maj = self.y_label_encoder.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        return X

# base voting classifier with some extra method(s)
class VotingClassifier(BaseVotingClassifier):

    # first dimension must be samples for `cross_val_predict`
    def _collect_probas_flatten(self, X):
        return np.concatenate(self._collect_probas(X), axis = 1)

# voting classifier which determines optimal weights
class VotingClassifierWeightTune(_BaseComposition, ClassifierMixin, TransformerMixin):

    def __init__(self, estimators, scoring, weights_granularity = 10, weights_cv = 3, voting = 'soft', n_jobs = 1, flatten_transform = None):
        self.voter = VotingClassifier(estimators = estimators, voting = voting, n_jobs = n_jobs, flatten_transform = flatten_transform)
        self.scorer = check_scoring(self.voter, scoring = scoring)
        self.voting = voting
        self.weights_granularity = weights_granularity
        self.weights_cv = weights_cv
        self.weights_ = None

        if voting == 'hard' and self.scorer.__class__ != _PredictScorer:
            raise Exception("When voting = 'hard', must use scorer that doesn't need a threshold or probability input (e.g. accuracy)")

    @property
    def weights(self):
        return self.weights_

    @weights.setter
    def weights(self, w):
        self.voter.weights = w
        self.weights_ = w

    def fit(self, X, y, sample_weight = None):
        # create universe of weights to test
        n_est = len(self.voter.estimators)
        weights = [[j/(self.weights_granularity * 1.0) for j in i] for i in stars_and_bars(n_est, self.weights_granularity)]

        # compute predictions from each model once for efficiency
        # NOTE: chose to make VotingClassifier an object within VotingClassifierWeightTune b/c
        # o.w. `cross_val_predict` would create a recursion
        func = '_collect_probas_flatten' if self.voting == 'soft' else '_predict'
        est_pred = cross_val_predict(self.voter, X, y, cv = self.weights_cv, method = func)

        # score with different weights
        # NOTE: not using `scorer` conventially because not passing it an estimator
        y_le = LabelEncoder().fit(y)
        sept = ScoreEstPassThru(y_le)
        scores = []
        for w in weights:
            if self.voting == 'soft':
                pseudo_X = np.average(
                    np.split(est_pred, n_est, axis = 1),
                    axis = 0, weights = w)
            else:
                pseudo_X = np.apply_along_axis(
                    lambda x: np.bincount(x, weights = w, minlength = len(y_le.classes_)),
                    axis = 1, arr = est_pred)
            score = self.scorer(sept, pseudo_X, y)
            scores.append(score)

        # determine optimal weight
        optimal_weights = weights[np.argmax(scores)]
        self.weights = optimal_weights

        # tune the estimators on full data
        self.voter.fit(X, y, sample_weight)

    def transform(self, X):
        return self.voter.transform(X)

    def predict(self, X):
        return self.voter.predict(X)

    def predict_proba(self, X):
        return self.voter.predict_proba(X)

    def set_params(self, **params):
        self.voter._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        return self.voter._get_params('estimators', deep = deep)

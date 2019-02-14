"""Testing Configuration File.

Alters the behavior of all testing modules.

Attributes:
    data_examples (:obj:`callable`): Provides generators of example learning
        data.
    testing_defaults (:obj:`*`): Defines all default values needed by
        model-related modules.
    verbosity (int): Verbosity to use during unit tests. See `unittest`.
    _data_generator (callable): Returns learning-ready data according to the
        given dataset, learning rate, decay rate, title, and target label.
    _lclass (callable): Loads learning-ready data for linear classification
        (i.e. `sklearn`'s breast cancer DB).
    _lreg (callable): Loads learning-ready data for linear regression (i.e.
        `sklearn`'s Boston housing prices DB).

"""
import numpy as _np

from sklearn.datasets import load_boston as _load_boston, \
                             load_breast_cancer as _load_breast_cancer

def _data_generator(dataset, lr, dr, t, t_label, classes=None):
    return (_np.matrix(dataset.data), _np.matrix(dataset.target).T, t,
            list(dataset.feature_names), t_label,
            dict(decay_rate=dr, learning_rate=lr), classes)

_lreg = lambda: _data_generator(_load_boston(), 5e-2, 0.3,
                                "Boston Housing Prices", "MEDV")

_lclass = lambda: _data_generator(_load_breast_cancer(), 5e-2, 0.3,
                                  "Breast Cancer", "Classes",
                                  classes=["Has breast cancer",
                                           "No breast cancer"])

data_examples = {
    "Linear": {
        "classification": _lclass,
        "regression": _lreg
    }
}

testing_defaults = {
    "zero_cutoff": 1e-6  #: Largest value to treat as zero.
}

verbosity = 1

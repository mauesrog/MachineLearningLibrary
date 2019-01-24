"""Learner Module.

Abstracts all training and testing from the actual mathematical models.

Attributes:
    SGD_DEFAULT_K (int): Default number of buckets to use in k-bucket SGD.

"""
from matplotlib import pyplot as plot
from time import sleep

from models.linear import LinearModel
from utils.stats import batches

import numpy as np

DEFAULT_DECAY_RATE = 0.5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHES = 1e4
MIN_DELTA_EPOCH = 1e-6
SGD_DEFAULT_K = 50

models = dict(linear=LinearModel)

class Learner():
    """Learner Module.

    Defines methods to train, evaluate, and test mathematical models that
    implement the abstract class `Model`.

    """
    def __init__(self):
        """Learner Constructor.

        """
        pass

    def gradient_checker(self, regularization, type, **kwargs):
        grad_norms, ngrad_norms = self._get_model(type)(regularization).gradient_checker(*kwargs.values())
        self.plot("Gradient checker", real=grad_norms, numerical=ngrad_norms)

    def plot(self, title, **kwargs):
        """Model Plotter.

        Plots the given vectors and labels them accordingly.

        Example:
            The following would plot the observations and predictions of only
            two data points:

            >>> Learner.plot(observations=[0.1, 0.2], predictions=[0.11, 0.3])

        Args:
            title (str): Plot display title.
            **kwargs: Dictionary where keys represent y-axis labels and values
                represent y-axis values e.g. .

        Raises:
            ValueError: If data vectors disagree in length.

        """
        X_axis = None
        """float: Values for the horizontal axis."""

        for label, vals in kwargs.iteritems():
            if X_axis is None:
                X_axis = [float(i) for i in range(0, len(vals))]
            elif len(X_axis) != len(vals):
                raise ValueError('All data points must be of equal length.')

            plot.plot(X_axis, np.asarray(vals), label=label.capitalize())

        if X_axis:
            plot.title(title)
            plot.legend()
            plot.show()



    def train(self, raw_X, Y, type, k=None, plot=False,
              regularization=0.0, **kwargs):
        """Model Trainer.

        Args:
            X (np.matrix): Training feature set. Shape: n x d.
            Y (np.matrix): Training observation set. Shape: n x 1.
            type (str): Model name. Must match an existing implementation of
                `Model`.
            exact (bool, optional): `True` means to use the global minimum of
                the loss function, while `False` means to use stochastic
                methods. Defaults to `False`.
            plot (bool, optional): Whether to plot training predictions against
                training observations. Defaults to `False`.
            **kwargs: Additional arguments needed by the model and/or the
                training method.

        Returns:
            (tuple of np.matrix, float): Best parameters found according to
                best training error, which gets returned as well.

        Raises:
            ValueError: If no model exists with name `type`.

        """
        if type not in models:
            raise ValueError("Unkown model type: '%s'." % type)

        Model = self._get_model(type)
        model = Model(regularization)

        del model.params

        X = model.augment(raw_X)
        training_err = self._cross_validate(X, Y, model, plot=plot, k=k,
                                            **kwargs)

        return lambda A: model.predict(model.augment(A)), training_err

    def _cross_validate(self, X, Y, model, k=SGD_DEFAULT_K, plot=False, exact=False,
                        **kwargs):
        """K-Fold Cross Validation.

        Trains the given model using stochastic gradient descent (SGD) via `k`
        buckets.

        Args:
            X (np.matrix): Training feature set. Shape: n x d.
            Y (np.matrix): Training observation set. Shape: n x 1.
            model (Model): Mathematical model to train.
            k (int, optional): Number of data points per bucket. Defaults to
                `SGD_DEFAULT_K`.
            plot (bool, optional): Whether to plot the training observations and
                predictions of all `k` buckets. Defaults to `False`.
            **kwargs: Additional arguments needed by the model for gradient
                computation.

        Returns:
            float: Training error (i.e. the average error across all k buckets).

        """
        buckets = batches(X, Y, k)
        err = []
        min_err = float("inf")
        optimal_params = None
        params = None

        for i in range(0, len(buckets)):
            test = buckets[i]
            train = None

            for j in range(0, len(buckets)):
                if i != j:
                    if train is None:
                        train = buckets[j]
                    else:
                        train = np.matrix(np.concatenate((train, buckets[j])))

            train_X = np.matrix(train[:, :-1])
            train_Y = np.matrix(train[:, -1])

            test_X = np.matrix(test[:, :-1])
            test_Y = np.matrix(test[:, -1])

            params = self._train_helper(model, train_X, train_Y, params=params, exact=exact, **kwargs)
            training_err, Y_hat = model.evaluate(test_X, test_Y, params=params, regularize=False)

            if plot:
                self.plot("Training results bucket %d. Training error: %f" % (i, training_err), observations=test_Y,
                          predictions=Y_hat)

            err.append(training_err)

            if training_err < min_err:
                min_err = training_err
                optimal_params = params

        if optimal_params is not None:
            model.train(X, Y, exact=True)

            model.params = optimal_params

        return np.mean(err)

    def _train_helper(self, model, train_X, train_Y, exact=False, **kwargs):
        err = model.train(train_X, train_Y, exact=exact)

        if exact:
            params = model.params
        else:
            params = self._sgd(model, train_X, train_Y, **kwargs)

        return params

    def _sgd(self, model, X, Y, learning_rate=DEFAULT_LEARNING_RATE,
             decay_rate=DEFAULT_DECAY_RATE, max_epoches=DEFAULT_MAX_EPOCHES,
             **kwargs):

        epoch = 0
        err = []
        lr = learning_rate
        params = model.params

        while epoch < max_epoches and not self._is_epoch_terminal(err):
            grads = model.gradient(X, Y, params=params)

            for i in range(0, len(params)):
                params[i] -= lr * grads[i]

            err.append(model.evaluate(X, Y, params=params)[0])
            epoch += 1

            lr /= (1.0 + epoch * decay_rate * lr ** (3 / 4))

        self.plot("Descent: %f" % err[-1], loss=err)

        return params

    def _get_model(self, type):
        if type not in models:
            raise ValueError("Unkown model type: '%s'." % type)

        return models[type]

    def _is_epoch_terminal(self, err):
        return len(err) > 1 and (abs(err[-1] - err[-2]) < MIN_DELTA_EPOCH or err[-1] > err[-2])

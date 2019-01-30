"""Learner Module.

Abstracts all training and testing from the actual mathematical models.

Attributes:
    DEFAULT_DECAY_RATE (float): Default positive rate at which SGD's update rule
        should descend.
    DEFAULT_DECAY_RATE (float): Default positive rate at which SGD's learning
        rate should decline.
    DEFAULT_MAX_EPOCHES (int): Default maximum number of epoches before
        interrupting the descent.
    DEFAULT_SGD_K (int): Default number of buckets to use in k-bucket SGD.
    MIN_DELTA_EPOCH (float): Smallest accepted difference in model loss from
        one epoch to the next.
    models (:obj:`Model`): All available learning models.

"""
from matplotlib import pyplot as plot
import numpy as np
from copy import deepcopy

from utils.stats import batches
from models.linear import LinearModel


DEFAULT_DECAY_RATE = 0.5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_EPOCHES = 1e4
DEFAULT_SGD_K = 50
MIN_DELTA_EPOCH = 1e-6

models = dict(Linear=LinearModel)


class _ModelWrapper(object):
    """Model Wrapper.

    Encapsulates all `Model` instances in `Learner` to provide a more simplified
    and protected learning API.

    Attributes:
        _learner (Learner): Handle to learner responsible for instantiating
            `self`.

    """
    def __init__(self, learner, Model):
        """Model Wrapper Constructor.

        """
        self._learner = learner
        self._model = Model()

    @property
    def model(self):
        """str: String representation of current model state."""
        return str(self._model)

    @model.setter
    def model(self, args):
        """Note: Only one flag can be active.

        Args:
            params (tuple of np.matrix, optional): Explicit model parameters.
            X (np.matrix): Feature set from which to infer the model parameters.
                Shape: n x d.

        Raises:
            AttributeError: If the flags `params` and `X` are both set or if
                neither is.

        """
        if ("params" in args and "X" in args) or \
           not ("params" in args or "X" in args):
            raise AttributeError("Provide either `parmas` or `X`, not both or "
                                 "neither.")

        if "params" in args:
            self._model.params = args["params"]

        if "X" in args:
            self._model.init_params(args["X"])

    def predict(self, X):
        """Model Predictor Wrapper.

        Args:
            X (np.matrix): Feature set. Shape: n x d.

        Returns:
            See `Model.predict`.

        """
        return self._model.predict(X)

    def train(self):
        """Model Parameter Trainer.

        """
        pass

class _LearnerMetaClass(type):
    def __new__(metaname, classname, baseclasses, attrs):
        self = type.__new__(metaname, classname, baseclasses, attrs)

        for name, model in models.items():
            setattr(self, name, _ModelWrapper(self, model))

        return self

class Learner(object):
    """Learner Module.

    Defines methods to train, evaluate, and test mathematical models that
    implement the abstract class `Model`.

    """
    __metaclass__ = _LearnerMetaClass

    def __init__(self):
        """Learner Constructor.

        """
        pass

    def __iter__(self):
        """Learner Iterator.

        Yields:
            (str, _ModelWrapper): The next model key and wrapper.

        """
        model_key_to_tuple = lambda k: (k, getattr(self, k))
        """callable: Maps model keys to tuples of that include a model
        wrapper."""

        for model in map(model_key_to_tuple, models.keys()):
            yield model

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

    def _cross_validate(self, X, Y, model, k=DEFAULT_SGD_K, plot=False, exact=False,
                        **kwargs):
        """K-Fold Cross Validation.

        Trains the given model using stochastic gradient descent (SGD) via `k`
        buckets.

        Args:
            X (np.matrix): Training feature set. Shape: n x d.
            Y (np.matrix): Training observation set. Shape: n x 1.
            model (Model): Mathematical model to train.
            k (int, optional): Number of data points per bucket. Defaults to
                `DEFAULT_SGD_K`.
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

    def _decay_rule(self, epoch, learning_rate, decay_rate):
        """SGD Decay Rule.

        Determines what conditions determine the decline rate of SGD's learning
        rate.

        Args:
            epoch (int): Current iteration number.
            learning_rate (float): Positive rate at which SGD's update rule
                is currently descending.
            decay_rate (float, optional): Positive rate at which `learning_rate`
                should decline.

        Returns:
            float: Decayed learning rate.

        """
        decay_ratio = 1.0 + epoch * decay_rate * learning_rate ** (3 / 4)
        """float: Ratio at which `learning_rate` will decline."""

        return learning_rate * decay_ratio ** -1.0

    def _get_model(self, type):
        if type not in models:
            raise ValueError("Unkown model type: '%s'." % type)

        return models[type]

    def _is_epoch_terminal(self, epoch, max_epoches, err):
        """SGD Terminality Check.

        Judges if an epoch is final. The given epoch is terminal if and only if
        at least one of the following conditions are met:

            1. Maximum number of epoches has been reached.
            2. Difference in loss between the last and second to last epoches
               is smaller than `MIN_DELTA_EPOCH`.
            3. Loss increased in the last epoch rather than decreased.

        Args:
            epoch (int): Current iteration number.
            max_epoches (int): Maximum number of epoches allowed.
            err (list of float): Model loss across all epoches.

        Returns:
            bool: `True` if SGD should be interrupted, `False` otherwise.

        """
        # Maximum number of epoches reached.
        if epoch >= max_epoches:
            return True

        # Not enough epoches to judge.
        if len(err) < 2:
            return False

        delta_epoch = compose(abs, np.subtract)(err[-1], err[-2])
        """float: Difference in loss between the last and second to last
        epoches."""

        return  delta_epoch < MIN_DELTA_EPOCH or err[-1] > err[-2]

    def _sgd(self, model, X, Y, learning_rate=DEFAULT_LEARNING_RATE,
             decay_rate=DEFAULT_DECAY_RATE, max_epoches=DEFAULT_MAX_EPOCHES):
        """Stochastic Gradient Descent.

        Uses the given model's parameter gradients to decrease the model's loss
        until a teminal condition is found. See `_is_epoch_terminal`.

        Args:
            model (Model): Model to optimize.
            X (np.matrix): Training feature set. Shape: n x d.
            Y (np.matrix): Training observation set. Shape: n x 1.
            learning_rate (float, optional): Positive rate at which the update
                rule should descend. Defaults to `DEFAULT_LEARNING_RATE`.
            decay_rate (float, optional): Positive rate at which `learning_rate`
                should decline. Defaults to `DEFAULT_DECAY_RATE`.
            max_epoches (int, optional): Maximum number of epoches before
                interrupting the descent. Defaults to `DEFAULT_MAX_EPOCHES`.

        Returns:
            tuple of np.matrix: Trained model parameters.

        """
        epoch = 0
        """int: Current iteration number."""
        err = []
        """list of float: Model loss across all epoches."""
        lr = learning_rate
        """float: Descent rate in current epoch."""
        params = model.params
        """tuple of np.matrix: Model parameters in current epoch."""

        while not self._is_epoch_terminal(err):
            grads = model.gradient(X, Y, params=params)
            """tuple of np.matrix: Gradients of current model parameters."""

            params, curr_err = self._update_rule(params, grads, lr)
            lr = self._decay_rule(epoch, lr, decay_rate)

            err.append(curr_err)
            epoch += 1


        self.plot("Descent: %f" % err[-1], loss=err)

        return params

    def _train_helper(self, model, train_X, train_Y, exact=False, **kwargs):
        err = model.train(train_X, train_Y, exact=exact)

        if exact:
            params = model.params
        else:
            params = self._sgd(model, train_X, train_Y, **kwargs)

        return params

    def _update_rule(self, params, grads, learning_rate):
        """SGD Update Rule.

        Determines how a model's parameters should be updated from one epoch to
        the next based on their gradients and the given learning rate.

        Args:
            params (tuple of np.matrix): Model parameters to update.
            grads (tuple of np.matrix): Gradients of `params`.
            learning_rate (float): Positive rate at which to descend.

        Returns:
            (tuple of np.matrix, float): Deep copy of model parameters plus
                its loss after the update.

        """
        new_params = deepcopy(params)
        """tuple of np.matrix: Parameter update."""

        for i in range(0, len(params)):
            new_params[i] -= learning_rate * grads[i]

        return new_params, model.evaluate(X, Y, params=new_params)[0]

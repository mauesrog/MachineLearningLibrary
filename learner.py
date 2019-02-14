"""Learner Module.

Abstracts all training and testing from the actual mathematical models.

Attributes:
    active_models (:obj:`Model`): All available learning models.

    See `config.models`.

"""
import numpy as np
from copy import deepcopy

from config import learner_defaults
from utils.general import compose
from utils.stats import batches, shuffle_batches
from utils.linalg import append_bottom
from utils.visualization import Visualization
from models.linear import LinearModel
from models.modelwrapper import ModelWrapper

DEFAULT_DECAY_RATE = learner_defaults["decay_rate"]
DEFAULT_LEARNING_RATE = learner_defaults["learning_rate"]
DEFAULT_MAX_EPOCHES = learner_defaults["max_epoches"]
DEFAULT_SGD_K = learner_defaults["sgd_k"]
MIN_DELTA_EPOCH = learner_defaults["min_delta_epoch"]

active_models = dict(Linear=LinearModel)


class _LearnerMetaClass(type):
    """Learner MetaClass.

    Performs abstractions to `Learner` in order to provide a cleaner API.

    """
    def __new__(metaname, classname, baseclasses, attrs):
        """Learner Instantiator.

        Adds ModelWrapper attributes to the learner for each currently
        implemented model.

        Args:
            See `type.__new__`.

        Returns:
            Learner: Augmented class instance.

        """
        self = type.__new__(metaname, classname, baseclasses, attrs)
        """Learner: Augmented class instance."""

        for name, model in active_models.items():
            setattr(self, name, ModelWrapper(model))

        return self

class Learner(object):
    """Learner Module.

    Defines methods to train, evaluate, and test mathematical models that
    implement the abstract class `Model`.

    Returns:
        (tuple of np.matrix, float): Best parameters found according to
            best training error, which gets returned as well.

    """
    __metaclass__ = _LearnerMetaClass

    def __init__(self):
        """Learner Constructor.

        """
        # Add a Reference to `self` for all `ModelWrapper` instances to train.
        for name, model in active_models.items():
            getattr(self, name)._update_learner(self)

    def __iter__(self):
        """Learner Model Iterator.

        Returns an iterator of `ModelWrapper` instances.

        Yields:
            (str, _ModelWrapper): The next model key and wrapper.

        """
        model_key_to_tuple = lambda k: (k, getattr(self, k))
        """callable: Maps model keys to tuples of that include a model
        wrapper."""

        for model in map(model_key_to_tuple, active_models.keys()):
            yield model

    @staticmethod
    def visualize(*args, **kwargs):
        """Visualization Wrapper.

        Returns a `Visualization` instance with the given number of subplots.

        Args:
            subplots (int): Number of subplots.

        Returns:
            Visualization: Dynamic plotter handle.

        """
        return Visualization(*args, **kwargs)

    def _cross_validate(self, X, Y, model, k=DEFAULT_SGD_K, **kwargs):
        """K-Fold Cross Validation.

        Creates as many buckets with `k` elements as possible from the given
        datasets and trains the model with all possible bucket permutations i.e.
        reserves a single bucket for testing and the rest for training.

        Args:
            X (np.matrix): Training feature set. Shape: n x d.
            Y (np.matrix): Training observation set. Shape: n x 1.
            model (Model): Mathematical model to train.
            k (int, optional): Number of data points per bucket. Defaults to
                `DEFAULT_SGD_K`.
            exact (bool, optional): `True` if training should be done
                analytically, or `False` if it should be done stochastically.
                Defaults to `False.`

        Returns:
            float: Training error (i.e. average testing error across all bucket
                permutations).

        """
        buckets = batches(X, Y, k)
        """list of np.matrix: Dataset batches with at least `k` data point
        each."""
        err = []
        """list of float: Testing errors of all permutations."""
        optimal_params = None
        """tuple of np.matrix: Parameters with the smallest testing error."""
        min_err = float("inf")
        """float: Testing error of best parameters encountered."""

        splitter = lambda dataset: (dataset[:, :-1], dataset[:, -1])
        """callable: Separates a dataset into feature and observation sets."""

        for i in range(len(buckets)):
            train = deepcopy(buckets)
            """list of np.matrix: Training buckets."""
            test_X, test_Y = compose(splitter, train.pop)(i)
            """(np.matrix, np.matrix): Testing feature and observation sets."""

            params = self._train_helper(model, train, **kwargs)
            """tuple of np.matrix: Trained model parameters."""
            training_err = model.evaluate(test_X, test_Y, params=params,
                                          regularize=False)[0]
            """float: Loss after training according to reserved testing
            bucket."""

            err.append(training_err)

            # Set current trained parameters as optimal if `training_err` is
            # smaller than all other training errors encountered before.
            if training_err < min_err:
                min_err = training_err
                optimal_params = params

        # Only update model parameters if a good match is found.
        if optimal_params is not None:
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

    def _sgd(self, model, datasets, learning_rate=DEFAULT_LEARNING_RATE,
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

        while not self._is_epoch_terminal(epoch, max_epoches, err):
            curr_err = 0.0
            """float: Training error averaged across all buckets."""

            for train_X, train_Y in datasets:
                grads = model.gradient(train_X, train_Y, params=params)
                """tuple of np.matrix: Gradients of current model parameters."""

                params = self._update_rule(params, grads, lr)

                curr_err += model.evaluate(train_X, train_Y, params=params)[0]

            curr_err /= len(datasets)

            lr = self._decay_rule(epoch, lr, decay_rate)

            err.append(curr_err)
            epoch += 1

        return params

    def _train_helper(self, model, train_buckets, exact=False, **kwargs):
        """Model Trainer Helper.

        Trains model parameters depending either analytically or numerically.

        Args:
            model (Model): Model to be trained.
            train_buckets (list of np.matrix): Stochastic and randomized
                representation of dataset.
            exact (bool, optional): `True` if training should be done
                analytically, `False` otherwise. Defaults to `False`.
            **kwargs: Options that define SGD's behavior.

        Returns:
            tuple of np.matrix: Trained model parameters.

        """
        concatenator = lambda A, B: append_bottom(A, B)
        """callable: Appends matrix `B` to the bottom of matrix `A`."""
        splitter = lambda dataset: (dataset[:, :-1], dataset[:, -1])
        """callable: Separates a dataset into feature and observation sets."""
        datasets = shuffle_batches(train_buckets)
        """list of np.matrix: Re-ordered batches."""

        model.init_params(train_buckets[0][:, :-1])

        if exact:
            train_X, train_Y = compose(splitter, reduce)(concatenator, datasets)
            """(np.matrix, np.matrix): Training feature and observation sets."""

            model.train(train_X, train_Y)
        else:
            model.params = self._sgd(model, map(splitter, datasets), **kwargs)

        return model.params

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

        for p, g in zip(new_params, grads):
            p -= learning_rate * g

        return new_params

"""Model Wrapper.

Contains ModelWrapper class.

Attributes:
    See `config.models`.

"""
from utils.general import compose
import numpy as np

from utils.stats import normalize


class ModelWrapper(object):
    """Model Wrapper.

    Encapsulates all `Model` instances in `Learner` to provide a more simplified
    and protected learning API.

    Attributes:
        _learner (Learner): Handle to learner responsible for instantiating
            `self`.
        _model (Model): Wrapped instance.

    """
    def __init__(self, Model):
        """Model Wrapper Constructor.

        Args:
            Model (Model): Class being wrapped.

        """
        self._learner = None
        self._model = Model()

    def __str__(self):
        """Model Wrapper Stringifier.

        Returns:
            str: Description of wrapped model's state.

        """
        return self.model

    @property
    def model(self):
        """str: String representation of wrapped model's state."""
        return str(self._model)

    @model.setter
    def model(self, args):
        """Note: Only one flag can be active.

        Args:
            args (:obj:`*`): Inludes a `params` key with explicit model
                parameters or an `X` key with a feature set from which to infer
                the model parameters.

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
            compose(self._model.init_params, self._model.augment)(args["X"])

    def evaluate(self, raw_X, Y):
        """Model Evaluation Wrapper.

        Args:
            raw_X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.

        Returns:
            float: Loss.

        """
        X = compose(normalize, self._model.augment)(raw_X)
        """np.matrix: Normalized and augmented feature set."""

        return self._model.evaluate(X, Y, regularize=False)[0]

    def predict(self, X):
        """Model Predictor Wrapper.

        Args:
            X (np.matrix): Feature set. Shape: n x d.

        Returns:
            See `Model.predict`.

        """
        return compose(self._model.predict, normalize, self._model.augment)(X)

    def train(self, raw_X, Y, **kwargs):
        """Model Parameter Trainer.

        Trains the wrapped model's parameters via analytical or numerical
        methods i.e. either exact global minimum computation or approximation
        via SGD.

        Args:
            raw_X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x 1.
            **kwargs: Additional flags that determine the learner's training
                behavior.

        Returns:
            float: Training error.

        """
        def action():
            """Train Update Action.

            Defines the routine to run after the feature sets and parameters
            have been validated.

            Returns:
                See `Learner._cross_validate`.

            """
            X = compose(normalize, self._model.augment)(raw_X)
            """np.matrix: Augmented feature set."""

            return self._learner._cross_validate(X, Y, self._model, **kwargs)

        return self._model._update_model(action, X=raw_X, Y=Y, no_params=True)

    def _update_learner(self, learner):
        """`Learner` Reference Creator.

        Saves a reference to the `Learner` responsible for instantiating `self`.

        Args:
            learner (Learner): Referenct to learner object.

        """
        self._learner = learner

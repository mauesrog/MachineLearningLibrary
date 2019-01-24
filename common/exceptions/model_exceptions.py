import numpy as _np

DEFAULT_MODEL_ERROR_MESSAGE = "Unknown reason."

class IncompleteModelError(Exception):
    """Incomplete Model Error.

    Raised when attempting to use a model that is missing data.

    Attributes:
        message (str): Description of error.

    """
    def __init__(self, message):
        """Invalid Model Parameters Error Constructor.

        Args:
            message (str): See class attributes.

        """
        self.message = message

        super(Exception, self).__init__(self.message or
                                        DEFAULT_MODEL_ERROR_MESSAGE)

class InvalidModelParametersError(Exception):
    """Invalid Model Parameters Error.

    Raised when a model encounters invalid parameters.

    Attributes:
        message (str): Description of error.
        params (tuple): Invalid model parameters.

    """
    def __init__(self, params, reason=None):
        """Invalid Model Parameters Error Constructor.

        Args:
            params (tuple): See class attributes.
            isType (bool): Whether it's a type-related error.
            reason (str, optional): Description of what is wrong with the
                parameters. Defaults to `None`, which will trigger a default
                error message.

        """
        self.params = params
        self.message = reason

        if type(params) != tuple:
            self.message = "Expected 'tuple of matrix', saw '%s' instead." \
                           % type(params).__name__
        else:
            for p in params:
                if type(p) != _np.matrix:
                    self.message = "Expected 'matrix' parameters, saw '%s' " \
                                   "instead." % type(p).__name__
                    break

                if p.size == 0:
                    self.message = "Parameters cannot be empty!"
                    break

        super(Exception, self).__init__(self.message or
                                        DEFAULT_MODEL_ERROR_MESSAGE)

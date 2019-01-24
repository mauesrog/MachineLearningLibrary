DEFAULT_DATA_SET_ERROR_MESSAGE = "Unknown reason."


class IncompatibleDataSetsError(Exception):
    """Incompatible Data Sets Error.

    Raised when attempting to do an operation between two matrices of
    incongruous dimensions.

    Attributes:
        message (str): Description of error.
        operation (str): Lower-case noun describing the attempted action that
            caused the error.
        sets ((np.matrix, np.matrix)): The two incompatible datasets.

    """
    def __init__(self, X, Y, operation):
        """Incompatible Data Sets Error Constructor.

        Args:
            X (np.matrix): First dataset in incompatible pair.
            Y (np.matrix): Second dataset in incompatible pair.
            operation (str): See class attributes.

        """
        self.message = "Matrices of shapes %s and %s are incompatible with " \
                       "%s." % (X.shape, Y.shape, operation)
        self.operation = operation
        self.sets = X, Y

        super(Exception, self).__init__(self.message)

class InvalidDataSetError(Exception):
    """Invalid Data Set Error.

    Raised when there is something wrong with either a feature or an observation
    dataset.

    Attributes:
        message (str): Description of error.
        set (np.matrix): Faulty dataset.

    """
    def __init__(self, X, isType=False, reason=DEFAULT_DATA_SET_ERROR_MESSAGE):
        """Invalid Data Set Error Constructor.

        Args:
            X (np.matrix): Faulty dataset.
            isType (bool): Whether it's a type-related error.
            reason (str, optional): Description of what is wrong with data set.
                Defaults to `DEFAULT_DATA_SET_ERROR_MESSAGE`.

        """
        self.set = X
        self.message = reason

        if isType:
            self.message = "Expected 'matrix', saw '%s'." % type(X).__name__


        super(Exception, self).__init__(self.message)

class InvalidFeatureSetError(InvalidDataSetError):
    """Invalid Feature Set Error.

    Raised when encountering a faulty feature set.

    """
    def __init__(self, X, **kwargs):
        """Invalid Feature Set Error Constructor.

        Args:
            X (np.matrix): Faulty feature set.
            **kwargs: Aditional `InvalidDataSetError` flags.

        """
        super(InvalidFeatureSetError, self).__init__(X, **kwargs)

class InvalidObservationSetError(InvalidDataSetError):
    """Invalid Observation Set Error.

    Raised when encountering a faulty observation set.

    """
    def __init__(self, X, **kwargs):
        """Invalid Observation Set Error Constructor.

        Args:
            X (np.matrix): Faulty observation set.
            reason (str): Description of what is wrong with observation set.

        """
        super(InvalidObservationSetError, self).__init__(X, **kwargs)

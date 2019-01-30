"""Utilities Configuration File.

Alters the behavior of utility modules.

Attributes:
    linalg_defaults (:obj:`*`): Defines default values for all optional values
        in any linear algebra module.

"""
linalg_defaults = {
    "max_random_value": 100.0,  #: Default maximum random number allowed.
    "min_random_value": 0.0  #: Default minimum random number allowed.
}

"""Utilities Configuration File.

Alters the behavior of utility modules.

Attributes:
    linalg_defaults (:obj:`*`): Defines default values for all optional values
        in any linear algebra module.
    visualization_defaults (:obj:`*`): Defines default values for all optional
        values in `Visualization`.

"""
linalg_defaults = {
    "max_random_value": 100.0,  #: Default maximum random number allowed.
    "min_random_value": 0.0  #: Default minimum random number allowed.
}

visualization_defaults = {
    "bottom": 0.1,  #: Default bottom padding of figures.
    "figsize": (12.94, 8.0),  #: Default figure size.
    "left": 0.06,  #: Default left padding of figures.
    "margins": (0.1),  #: Default subplot margins.
    "markeredgewidth": 0.0,  #: Default width for all marker edges.
    "numpoints": 1,  #: Default number of points per legend.
    "right": 0.97,  #: Default right padding of figures.
    "title_fontsize": 16,  #: Default font size for figure title.
    "top": 0.91  #: Default top padding of figures.
}

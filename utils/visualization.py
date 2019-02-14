"""Visualization Module.

Defines the `Visualization` class.

Attributes:
    See `config.models`.

"""
from matplotlib import pyplot as plt
import numpy as np
from random import choice

from config.utils import visualization_defaults
from utils.general import compose
from utils.stats import validate_datasets
from models.modelwrapper import ModelWrapper

DEFAULT_BOTTOM_PADDING = visualization_defaults["bottom"]
DEFAULT_FIGURE_SIZE = visualization_defaults["figsize"]
DEFAULT_LEFT_PADDING = visualization_defaults["left"]
DEFAULT_MARGINS = visualization_defaults["margins"]
DEFAULT_MARKER_EDGE_WIDTH = visualization_defaults["markeredgewidth"]
DEFAULT_NUMPOINTS = visualization_defaults["numpoints"]
DEFAULT_RIGHT_PADDING = visualization_defaults["right"]
DEFAULT_TITLE_FONT_SIZE = visualization_defaults["title_fontsize"]
DEFAULT_TOP_PADDING = visualization_defaults["top"]


class Visualization(object):
    """Visualization Class.

    Provides an interface that simplifies all data visualization operations,
    such as plotting observations and predictions.

    """
    def __init__(self, title, subplots, **kwargs):
        """Visualization Constructor.

        Args:
            title (str): Plot display label.
            subplots ((int, int)): Determines the number of subplots to display
                in the form of rows and columns.
            **kwargs: Additional arguments to supply
                `matplotlib.pyplot.subplots`.

        """
        self._fig, all = plt.subplots(*subplots, figsize=DEFAULT_FIGURE_SIZE,
                                      **kwargs)
        self._shape = subplots

        self._fig.suptitle(title, fontsize=DEFAULT_TITLE_FONT_SIZE)

        try:
            self._ax = []

            for sub in all:
                try:
                    for ax in sub:
                        self._ax.append(ax)
                except:
                    self._ax.append(sub)
        except:
            self._ax = [all]


    @staticmethod
    def show():
        """Show Wrapper.

        Wraps `matplotlib.pyplot.show`.

        """
        plt.show()

    @staticmethod
    def close(figs="all"):
        """Close Wrapper.

        Wraps `matplotlib.pyplot.close`.

        """
        plt.close(figs)

    @staticmethod
    def _generate_color():
        """Color Generator.

        Generates random color.

        Returns:
            (float, float, float): Color in RGB format with scale [0, 1].

        """
        levels = range(32, 256, 32)
        """int: Possible RGB values."""

        rgb = [choice(levels) for i in range(3)]
        """(int, int, int): RGB color with scale [0, 256]."""

        return compose(tuple, map)(lambda v: v / 255.0, rgb)

    def plot_features(self, X, Y, features, model, names=None, ylabel="Values"):
        """Feature Plotter.

        Displays a scatter plot of the provided features' predictions and the
        given observations.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x d.
            features (list of int): Column numbers of target features.
            model (ModelWrapper): Model to use for predictions.

        Returns:
            callable: Handle to close all plots.

        """
        if features is None:
            features = range(X.shape[1])

        if len(self._ax) < len(features):
            raise ValueError("%d subplots are not enough to plot %d "
                             "features." % (len(self._ax), len(features)))

        for i, feature in enumerate(features):
            x, values, error = self._plot_feature(X, Y, feature, model)
            """(list of float, :obj:`list of float`): Feature's x and y
            values to plot."""

            kwargs = dict(xlabel="Feature %d" % feature if names is None \
                                                        else names[feature])

            if i % self._shape[1] == 0:
                kwargs["ylabel"] = ylabel

            lines = self._subplot(i, x, values, legend=False, **kwargs)
            regression_lines = self._best_fit_lines(x, values, i)

            if i % self._shape[1] != 0:
                self._ax[i].yaxis.set_ticks([])

        map(self._empty_subplot, range(len(features), len(self._ax)))

        padding = {
            "bottom": DEFAULT_BOTTOM_PADDING,
            "left": DEFAULT_LEFT_PADDING,
            "right": DEFAULT_RIGHT_PADDING,
            "top": DEFAULT_TOP_PADDING
        }

        labels = "Observations", "Predictions", "True line", "Prediction line"

        self._fig.subplots_adjust(hspace=0.3, wspace=0.0, **padding)
        self._fig.legend(lines + regression_lines, labels, numpoints=1)
        self._fig.set_facecolor("white")

        title = self._fig._suptitle.get_text()

        self._fig.suptitle("%s: %.3f (MSE)" % (title, error))

        return lambda: plt.close('all')

    def _best_fit_lines(self, raw_x, values, subplot):
        """Best-Fit Line Plotter.

        Plots the lines of best fit according to the given x- and y-values.

        Args:
            raw_x (list of float): X-values with possible duplicates.
            values (:obj:`list of float`): Contains all y-values.
            subplot (int): Index of suplot to use for plotting.

        Return:
            list of matplotlib.lines.Line2D: Best-fit lines.

        """
        x = compose(list, np.unique)(raw_x)
        """list of float: X-values."""
        ax = self._ax[subplot]
        """matplotlib.axes.Axes: Subplot."""

        lines = []
        """list of matplotlib.lines.Line2D: Best-fit lines."""

        try:
            if len(values) == 0:
                raise ValueError("No y-values provided.")

            for i, (label, y) in enumerate(values.items()):
                color = self._generate_color()
                """(float, float, float): RGB color."""
                linestyle = "-" if label == "observations" else "--"
                """str: Plot linestyle. See `matplotlib.plot.pyplot`."""

                l, = ax.plot(x, compose(np.poly1d, np.polyfit)(raw_x, y, 1)(x),
                             color=color, linestyle=linestyle)
                """matplotlib.lines.Line2D: Best-fit model plotted last."""

                lines.append(l)
        except AttributeError:
            raise TypeError("Expected 'dict', saw '%s' instead." %
                            type(values).__name__)

        return lines

    def _empty_subplot(self, i):
        """Subplot Emptier.

        Empties the subplot matching the given index.

        Args:
            i (int): Subplot index.

        """
        ax = self._ax[i]
        """matplotlib.axes.Axes: Subplot."""

        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])

    def _plot_feature(self, X, Y, feature, model):
        """Feature Plotter.

        Displays a scatter plot of the provided feature' predictions and the
        given observations.

        Args:
            X (np.matrix): Feature set. Shape: n x d.
            Y (np.matrix): Observation set. Shape: n x d.
            feature (int): Column number of target feature.
            model (ModelWrapper): Model to use for predictions.

        Returns:
            (list of float, :obj:`list of float`, float): The x and y values to
                plot, plus prediction error.

        Raises:
            IndexError: If `feature` is out of range.
            TypeError: If `feature` is not an integer.
            ValueError: If `feature` is negative or if `model` is not an
                instance of `ModelWrapper`.

        """
        validate_datasets(X, Y)

        if not isinstance(feature, int):
            raise TypeError("Expected 'int', saw '%s' instead." %
                            type(feature).__name__)

        if not isinstance(model, ModelWrapper):
            raise TypeError("Expected 'ModelWrapper', saw '%s' instead." %
                            type(model).__name__)

        if feature < 0:
            raise ValueError("Negative feature indices are not allowed.")

        if feature >= X.shape[1]:
            raise IndexError("Feature index '%d' is out of range." % feature)

        aslist = lambda A: A.T[0, :].tolist()[0]
        """callable: Maps vectors to lists."""

        x = aslist(X[:, feature])
        """list of float: Feature's x values."""

        Y_hat = model.predict(X)
        """np.matrix: Predicitons."""
        y = map(aslist, [Y, Y_hat])
        """list of (list of float): Feature's y raw values."""
        values = compose(dict, zip)(["observations", "predictions"], y)
        """:obj:`list of float`: Feature's x and y values."""

        return x, values, model.evaluate(X, Y)

    def _subplot(self, i, x, values, title=None, xlabel=None, ylabel=None,
                 legend=True):
        """Subplotter.

        Plots the given data to the provided subplot.

        Args:
            i (int): Subplot index.
            x (list of float): X values.
            values (:obj:`list of float`): Dictionary where keys represent plot
                labels and values represent y values.
            title (str, optional): Subplot display name or `None` if no title
                should be displayed. Defaults to `None`.
            xlabel (str, optional): X-axis display title or `None` if no title
                should be displayed. Defaults to `None`.
            ylabel (str, optional): Y-axis display title or `None` if no title
                should be displayed. Defaults to `None`.

        Return:
            list of matplotlib.lines.Line2D: Plotted lines.

        """
        capitalize = lambda s: s.capitalize()
        """callable: Maps strings to capitalized strings."""

        ax = self._ax[i]
        """matplotlib.axes.Axes: Subplot."""

        lines = []
        """list of matplotlib.lines.Line2D: Plotted lines."""

        for i, (label_raw, y) in enumerate(values.iteritems()):
            label = compose(" ".join, map)(capitalize, label_raw.split("_"))
            """str: Capitalized, separeted version of `label_raw`."""
            color = self._generate_color()
            """(float, float, float): RGB color."""
            marker = "o" if label_raw == "observations" else "*"
            """str: Matplotlib marker style."""

            l, = ax.plot(x, y, color=color, marker=marker, linestyle="None",
                         label=label, markeredgewidth=DEFAULT_MARKER_EDGE_WIDTH)
            """matplotlib.lines.Line2D: Best-fit model plotted last."""

            lines.append(l)

        # Display title (if provided)
        if title is not None:
            ax.set_title(title)

        # Display axis labels (if provided)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        # Remove ticks from both the right y-axis and the top x-axis.
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.margins(DEFAULT_MARGINS)

        if legend:
            ax.legend(loc="best", numpoints=DEFAULT_NUMPOINTS)

        return lines

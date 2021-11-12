import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import seaborn as sns;




def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_spatio_temporal_data(x, figsize=(20, 8), save_fig=False, fig_name=None, mask=None):
    """x is a (T, n, n) tensor, T is the temporal dimension, nxn is the spatio dimension"""
    T = x.shape[0]
    n = x.shape[1]

    # each row can at most have 5 images
    if T % 5 == 0:
        fig, axs = plt.subplots(T // 5, 5, figsize=figsize)
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots((T // 5 + 1), 5, figsize=figsize)
        axs = axs.ravel()


    # row_label = np.arange(n)
    # col_label = np.arange(n)

    # for i in range(T):
    #     im, cbar = heatmap(x[i, ...], row_label, col_label, ax=axs[i], cmap='YlGn')
    #     if add_text:
    #         texts = annotate_heatmap(im, valfmt="{x:.1f}", **textkw)
    #
    # fig.tight_layout()
    # return fig
    if mask is not None:
        for i in range(T):
            if mask[i] == 1:
                sns.heatmap(x[i, ...], ax=axs[i], cbar=False)
            else:
                sns.heatmap(x[i, ...], ax=axs[i], cbar=False)
                for spine in axs[i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(6)
                    spine.set_visible(True)



    else:
        for i in range(T):
            sns.heatmap(x[i, ...], ax=axs[i], cbar=False)

    if save_fig:
        plt.savefig(fig_name)
    else:
        plt.show()


def set_interval_value(x, a, b):
    # function that associate to a float x, a value encoding its position with respect to the interval [a, b]
    #  the associated values are 0, 1, 2 assigned as follows:
    if x <= a:
        return 0
    elif a < x <= b:
        return 1
    else:
        return 2

def data2color(x, y, a, b, c, d):
    # This function works only with a list of 9 bivariate colors, because of the definition of set_interval_value()
    # x, y: lists or 1d arrays, containing values of the two variables
    #  each x[k], y[k] is mapped to an int  value xv, respectively yv, representing its category,
    # from which we get their corresponding color  in the list of bivariate colors
    if len(x) != len(y):
        raise ValueError('the list of x and y-coordinates must have the same length')
    n = 3
    xcol = [set_interval_value(v, a, b) for v in x]
    ycol = [set_interval_value(v, c, d) for v in y]
    idxcol = [int(xc + n*yc) for xc, yc in zip(xcol,ycol)]# index of the corresponding color in the list of bivariate colors

    return idxcol


def plot_spatio_temporal_data_with_uncertainty(x, figsize=(20, 8)):
    """x is a (T, n, n) tensor, T is the temporal dimension, nxn is the spatio dimension"""
    T = x.shape[0]
    n = x.shape[1]

    # each row can at most have 5 images
    if T % 5 == 0:
        fig, axs = plt.subplots(T // 5, 5, figsize=figsize)
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots((T // 5 + 1), 5, figsize=figsize)
        axs = axs.ravel()

    # create a bivariate color map
    jstevens = ["#e8e8e8", "#ace4e4", "#5ac8c8", "#dfb0d6", "#a5add3",
                "#5698b9", "#be64ac", "#8c62aa", "#3b4994"]  # use the existing colors
    #values = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    values = np.arange(9)

    cmap = ListedColormap(jstevens)

    print(cmap(values))

    for i in range(T):
        sns.heatmap(x[i, ...], cmap=cmap, ax=axs[i], cbar=False)  # don't draw a color bar


    plt.show()



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import mplcursors
from embed_documents import load_Y
import numpy as np


def plot_tsne(X, n_components, corpus, Xt=None, **kwargs):
    assert n_components in {2, 3}
    if not Xt:
        Xt = TSNE(n_components=n_components, **kwargs).fit_transform(X) 
#     if not kwargs:
        
    fig = plt.figure(figsize=(7, 7))
#     cmap = cm.get_cmap('RdBu')
    cmap = cm.get_cmap('Set1')
    Y, le = load_Y(corpus)
    kwargs = dict()
    if n_components == 2:
        ax = plt.axes()
        x = Xt[:, 0], Xt[:, 1]
    else:
        ax = plt.axes(projection='3d')
        x = Xt[:, 0], Xt[:, 1], Xt[:, 2]
        kwargs['depthshade'] = True
    scatter = ax.scatter(*x, c=Y, s=50, cmap=cmap, vmin=-.2, vmax=1.2, alpha=0.8, **kwargs)
    # Set up cursor clicking
    cursor = mplcursors.cursor(scatter)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(corpus.title(corpus[sel.target.index]))
    )
    
    # Custom legend
    kwargs = dict(marker = 'o', linestyle="None")
    custom_lines = [Line2D([0], [0], color='blue', **kwargs),
                    Line2D([0], [0], color='pink', **kwargs)]
    labels = map(lambda cat: cat.split(':')[1], le.inverse_transform([0, 1]))
    ax.legend(custom_lines, labels)
    plt.title("TSNE")
    plt.show()

    return Xt

def plot_logistic_regression(lr, Xt, corpus):

    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes()
    cmap = cm.get_cmap('RdBu')
    Y, le = load_Y(corpus)
    
    # Contour
    w = 2.5
    ws = 0.05
    xx, yy = np.mgrid[-w:w:ws, -w:w:ws]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = lr.predict_proba(grid)[:, 1].reshape(xx.shape)
    contour = ax.contourf(xx, yy, probs, 25, cmap=cmap,
                          vmin=0, vmax=1, alpha=0.8)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    # True points
    x1, x2 = Xt[:, 0], Xt[:, 1]
    scatter = ax.scatter(x1, x2, c=Y.flatten(), s=50, cmap=cmap, vmin=-.2, vmax=1.2, alpha=0.7, edgecolor='white')
    cursor = mplcursors.cursor(scatter)
    cursor.connect(
        "add", lambda sel: sel.annotation.set_text(corpus.title(corpus[sel.target.index]))
    )

    # Custom legend
    kwargs = dict(marker = 'o', linestyle="None")
    custom_lines = [Line2D([0], [0], color='blue', **kwargs),
                    Line2D([0], [0], color='red', **kwargs)]
    labels = map(lambda cat: cat.split(':')[1], le.inverse_transform([1, 0]))
    ax.legend(custom_lines, labels, loc='lower left')

    plt.title("Logistic regression decision boundary on TSNE embeddings")
    plt.show()
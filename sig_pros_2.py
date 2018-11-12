import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from pygsp import graphs, filters, plotting

G = graphs.Minnesota()
# G.coords.shape # co ords are the nodes can be visiualised

# fig, axes = plt.subplots(1, 2)
# _ = axes[0].spy(G.W, markersize=0.5)     # visualise the adjacency matrix in a spy plot
# G.plot(ax=axes[1])                       # visualise the graph in 2D coordinates
# plt.show()

G.compute_laplacian('combinatorial')
fig, axes = plt.subplots(1, 2)
axes[0].spy(G.L, markersize=5)
axes[1].hist(G.L.data, bins=50, log=True)

plt.show()

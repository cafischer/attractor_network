import matplotlib.pyplot as pl
import numpy as np
from brian2 import *
pl.style.use('paper')

start_scope()

n_neurons = 9**2
assert np.sqrt(n_neurons).is_integer()
n_neurons_row = n_neurons_column = int(np.sqrt(n_neurons))
neuron_spacing = 1

tau = 10*msecond

neuron_eqs = '''
ds/dt = - s / tau : 1
# Neuron position in space
x : 1 (constant)
y : 1 (constant)
'''

G = NeuronGroup(n_neurons, model=neuron_eqs)
idx = 0
for j in range(n_neurons_column):
    for i in range(n_neurons_row):
        G.x[idx] = (i - (n_neurons_row-1)/2.) * neuron_spacing
        G.y[idx] = (j - (n_neurons_column-1)/2.) * neuron_spacing
        idx += 1

# # plot positioning
# pl.figure()
# for i in range(n_neurons):
#     pl.scatter(G.x[i], G.y[i], color='k')
# pl.xlabel('Neuron x position')
# pl.ylabel('Neuron y position')
# pl.tight_layout()
# pl.show()

# Local connections Mexican hat style
S = Synapses(G, G, 'w : 1')
S.connect(condition='i!=j')
# Weight varies with distance
a = 1
beta = 3. / 13**2
gamma = 1.05 * beta
def w_fun(pos_vec):
    return a * np.exp(-gamma * np.linalg.norm(pos_vec)**2) - np.exp(-beta * np.linalg.norm(pos_vec)**2)

l = 1  # TODO
direction = 0  # TODO
for i in range(n_neurons):
    for j in range(n_neurons):
        S.w[i, j] = w_fun(np.array([G.x[i], G.y[i]]) - np.array([G.x[j], G.y[j]]) - l*direction)

# plot connections for one neurons
cmap = pl.cm.get_cmap('hot').reversed()
pl.figure()
i = (n_neurons-1)/2  # take neuron in the middle as reference
for j in range(n_neurons):
    S_min = np.min(S.w)
    S_max = np.max(S.w - S_min)
    sc = pl.scatter(G.x[j], G.y[j], color=cmap((S.w[i, j] - S_min) / S_max))
#pl.colorbar(sc)
pl.xlabel('Neuron x position')
pl.ylabel('Neuron y position')
pl.tight_layout()
pl.show()
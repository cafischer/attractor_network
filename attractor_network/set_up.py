import matplotlib.pyplot as pl
import numpy as np
from brian2 import *
import matplotlib.animation as animation
pl.style.use('paper')


if __name__ == '__main__':
    start_scope()

    # other parameters
    possible_directions = np.array([np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])])  # East, South, West, North

    # create neurons
    n_neurons = 10**2
    assert np.sqrt(n_neurons).is_integer()
    n_neurons_row = n_neurons_column = int(np.sqrt(n_neurons))
    assert n_neurons_row % 2 == 0  # handy for assigning directions
    neuron_spacing = 1  # cm

    tau = 10*msecond

    alpha = 0.10315
    a0 = 4
    diameter_neural_sheet = n_neurons_row * neuron_spacing
    delta_r = diameter_neural_sheet

    x_velocity = 0  # TODO
    y_velocity = 0  # TODO

    neuron_eqs = '''
    ds/dt = (- s + s_w_tot_rect) / tau : 1
    s_w_tot_rect = (s_w_tot + b) * int((s_w_tot + b) > 0) + 0 * int((s_w_tot + b) <= 0) : 1
    s_w_tot : 1
    b = A * (1 + alpha * (x_direction * x_velocity + y_direction * y_velocity)) : 1
    A = 1 * int(sqrt(x**2+y**2) < diameter_neural_sheet - delta_r) 
    + exp(-a0 * ((sqrt(x**2+y**2) - diameter_neural_sheet + delta_r) / delta_r)**2) 
    * int(diameter_neural_sheet - delta_r <= sqrt(x**2+y**2)) * int(sqrt(x**2+y**2) <= diameter_neural_sheet) : 1
    # Neuron position in space
    x : 1 (constant)
    y : 1 (constant)
    # preferred direction
    x_direction : 1 (constant)
    y_direction : 1 (constant)
    '''

    G = NeuronGroup(n_neurons, model=neuron_eqs)

    for i in range(n_neurons):
        # initial activity
        G.s[i] = np.random.uniform(0, 1)  # TODO

    # direction of neurons
    for i in range(n_neurons_row):
        for j in range(n_neurons_column):
            if i % 2 == 1:
                if j % 2 == 0:
                    G.x_direction[i + j * n_neurons_row] = possible_directions[0][0]
                    G.y_direction[i + j * n_neurons_row] = possible_directions[0][1]
                else:
                    G.x_direction[i + j * n_neurons_row] = possible_directions[1][0]
                    G.y_direction[i + j * n_neurons_row] = possible_directions[1][1]
            else:
                if j % 2 == 0:
                    G.x_direction[i + j * n_neurons_row] = possible_directions[2][0]
                    G.y_direction[i + j * n_neurons_row] = possible_directions[2][1]
                else:
                    G.x_direction[i + j * n_neurons_row] = possible_directions[3][0]
                    G.y_direction[i + j * n_neurons_row] = possible_directions[3][1]

    # position of neurons
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

    # local connections Mexican hat style
    S = Synapses(G, G, model='''
    s_w_tot_post = s_pre * w : 1 (summed)
    w : 1''')
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

    # # plot connections for one neurons
    # cmap = pl.cm.get_cmap('hot').reversed()
    # pl.figure()
    # i = (n_neurons-1)/2  # take neuron in the middle as reference (only works if n_neurons is odd)
    # for j in range(n_neurons):
    #     S_min = np.min(S.w)
    #     S_max = np.max(S.w - S_min)
    #     sc = pl.scatter(G.x[j], G.y[j], color=cmap((S.w[i, j] - S_min) / S_max))
    # #pl.colorbar(sc)
    # pl.xlabel('Neuron x position')
    # pl.ylabel('Neuron y position')
    # pl.tight_layout()
    # pl.show()

    # recording
    M = StateMonitor(G, ['s', 's_w_tot'], record=True)

    # run simulation
    run(1*second)

    # plot activity
    # pl.figure()
    # for i in range(n_neurons):
    #     pl.plot(M.t / ms, M.s[i])
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Activity')
    # #pl.show()
    #
    # pl.figure()
    # for i in range(n_neurons):
    #         pl.plot(M.t / ms, M.s_w_tot[i])
    # pl.xlabel('Time (ms)')
    # pl.ylabel('s_w_tot')
    # #pl.show()

    # grid map
    cmap = pl.get_cmap('hot')

    def update_plot(i, color, scat):
        scat.set_color(cmap(color[:, i]))
        return scat,

    fig, ax = pl.subplots()
    im = ax.scatter(G.x, G.y, c=M.s[:, 0], cmap=cmap, vmin=0, vmax=1)
    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(len(M.t)),
                                  interval=100,  # interval=how fast changes in ms
                                  fargs=(M.s, im), blit=True)
    ax.set_xlabel('Position x (cm)')
    ax.set_ylabel('Position y (cm)')
    fig.colorbar(im, ax=ax, label='Activity')
    pl.tight_layout()
    pl.show()
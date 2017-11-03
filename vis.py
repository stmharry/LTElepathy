import matplotlib
matplotlib.use('qt4agg')

import matplotlib.pyplot as plot
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

num_subcarriers = 25 * 12
num_samples = 60
step = 100

P = np.loadtxt(
    '/home/stmharry/Library/srsLTE/build/signal-kinetics/log-20',
)

P = np.reshape(
    P[:step * num_samples * num_subcarriers],
    (step * num_samples, num_subcarriers),
)
P = P + np.expand_dims(np.unwrap(P[:, 0]) - P[:, 0], axis=1)
P = P[np.arange(0, step * num_samples, step)]

for n in range(num_samples):
    P[n] = np.unwrap(P[n])

P_mean = np.mean(P, axis=1)
P_std = np.std(P, axis=1)

np.savez(
    'data/20.npz',
    delta_t=0.001 * step,
    num_samples=num_samples,
    phase=P_mean,
)

# PLOT

if False:
    fig = plot.figure(figsize=(16, 8))

    x = np.arange(num_subcarriers)
    y = np.arange(num_samples)
    (X, Y) = np.meshgrid(x, y)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, P)
    ax.set_xlabel('# subcarrier')
    ax.set_ylabel('# subframe')
    ax.set_zlabel('Phase')

    ax = fig.add_subplot(1, 2, 2)
    ax.errorbar(y, P_mean, yerr=P_std)

    plot.show(block=False)

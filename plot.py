import matplotlib
matplotlib.use('qt4agg')

import matplotlib.pyplot as plot
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import unwrap_phase

name = '10cm-2ports-50PRBs'
delta_t = 1e-2

num_samples = 600
num_ports = 2
num_subcarriers = 50 * 12

step_samples = 1
step_subcarriers = 12

P = np.fromfile(
    '/home/stmharry/Library/srsLTE/build/signal-kinetics/log/{:s}'.format(name),
    dtype=np.float32,
)
P = P[:num_samples * num_ports * num_subcarriers]
P = np.reshape(P, (num_samples, num_ports, num_subcarriers))
P = np.transpose(P, (1, 0, 2))

P = P[
    slice(0, num_samples, step_samples),
    :,
    slice(0, num_subcarriers, step_subcarriers),
]

for num_port in range(num_ports):
    P[num_port] = unwrap_phase(P[num_port])

# PLOT

if True:
    x = np.arange(num_subcarriers / step_subcarriers)
    y = np.arange(num_samples / step_samples)
    (X, Y) = np.meshgrid(x, y)

    fig = plot.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, P[0], color='b')
    ax.plot_surface(X, Y, P[1], color='r')
    ax.set_xlabel('# subcarrier')
    ax.set_ylabel('# subframe')
    ax.set_zlabel('Phase')

    plot.show()

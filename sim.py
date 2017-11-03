import matplotlib
matplotlib.use('qt4agg')

import matplotlib.pyplot as plot
import numpy as np
import scipy.ndimage

from util import World, Particles

with np.load('data/gen.npz') as data:
    delta_t = data['delta_t']
    num_samples = data['num_samples']
    phase = data['phase']


world = World(
    delta_t=delta_t,
    wave_freq=900e6,
    sep_length=0.19,
)

host = Particles(
    world=world,
    r=(0.0, 0.0),
)

particles = Particles(
    world=world,
    num_particles=10000,
    communism_ratio=0.30,
    r=host.r,
    r_std=5.0,
    v=(0.2, 0.0),
    v_std=0.1,
)


map_size = 4.0
num_bins = 512
power = 0.3
local_size = 1.0
local_bins = int(local_size / map_size * num_bins)

fig = plot.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.plot(host.r[0, 0], host.r[0, 1], 'ro')

(est_line,) = ax.plot(0, 0, 'bo')

xedges = np.linspace(-map_size, map_size, num_bins)
yedges = np.linspace(-map_size, map_size, num_bins)
(hist, _, _) = np.histogram2d(particles.r[:, 0], particles.r[:, 1], bins=[xedges, yedges])

img = ax.imshow(np.power(hist.T, power), extent=[-map_size, map_size, -map_size, map_size], origin='lower', cmap='binary')

ax.set_xlim(-map_size, map_size)
ax.set_ylim(-map_size, map_size)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

ax = fig.add_subplot(1, 2, 2)
(phase_line,) = ax.plot([], [], 'b')

ax.set_xlim(0, num_samples * delta_t)
ax.set_ylim(-np.pi, np.pi)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Phase (radian)')

plot.show(block=False)
fig.canvas.draw()

for num_sample in range(num_samples):
    print('Time={:.4f} s'.format(num_sample * delta_t))

    particles.predict(
        host=host,
        r_std=None,
        v=(0.2, 0.0),
        v_std=0.05,
    )
    particles.update(
        host=host,
        phase=phase[num_sample],
        phase_spread=0.25,
        r_spread=5.0,
    )
    particles.resample()

    (hist, _, _) = np.histogram2d(particles.r[:, 0], particles.r[:, 1], bins=[xedges, yedges])
    (xs, ys) = np.where(np.logical_and.reduce([
        hist != 0,
        hist == scipy.ndimage.filters.maximum_filter(hist, size=local_bins),
    ]))

    est_line.set_data(xedges[xs], yedges[ys])
    img.set_data(np.power(hist.T, power))
    img.autoscale()

    phase_line.set_data(np.arange(num_sample) * delta_t, phase[:num_sample])

    fig.canvas.draw()
    fig.canvas.flush_events()

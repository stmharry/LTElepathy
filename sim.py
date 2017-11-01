import matplotlib
matplotlib.use('qt4agg')

import matplotlib.pyplot as plot
import numpy as np

with np.load('20.npz') as data:
    delta_t = data['delta_t']
    num_samples = data['num_samples']
    phase = data['phase'] + 0.3

dim = 2
map_width = 4.0
map_height = 4.0
sep_length = 0.19
speed_of_light = 2.99792458e8
wave_freq = 900e6
wave_number = (2 * np.pi * wave_freq) / speed_of_light


class Particles(object):
    def normal(self, mean, std=None):
        mean = np.asarray(mean)
        if mean.ndim == 1:
            mean = np.expand_dims(mean, axis=0)
        d = mean.shape[1]

        if std is None:
            return mean * np.ones((self.num_particles, d))
        else:
            std = np.asarray(std)

            if std.ndim == 1:
                std = std / np.sqrt(d) * np.ones(d)

            return np.random.normal(mean, std, (self.num_particles, d))

    def __init__(self,
                 num_particles=1,
                 resample_ratio=1.0,
                 r=None,
                 r_std=None,
                 v=None,
                 v_std=None,
                 a=None,
                 a_std=None):

        self.num_particles = num_particles
        self.resample_ratio = resample_ratio

        if a is None:
            self.a = None
        else:
            self.a = self.normal(a, a_std)

        if v is None:
            self.v = None
        else:
            self.v = self.normal(v, v_std)

        if r is None:
            self.r = None
        else:
            self.r = self.normal(r, r_std)

        self.p = np.array([1, 0]) # TODO

    def predict(self,
                host,
                r=None,
                r_std=None,
                v=None,
                v_std=None,
                a=None,
                a_std=None):

        # state: add noise for diffusion
        if a is not None:
            self.a = self.normal(a, a_std)
            v = self.v + delta_t * self.a

        if v is not None:
            self.v = self.normal(v, v_std)
            r = self.r + delta_t * self.v

        if r is not None:
            self.r = self.normal(r, r_std)

        # measurement: no noise
        self.theta = wave_number * (
            np.linalg.norm((self.r - host.r) + sep_length / 2 * self.p, axis=1) -
            np.linalg.norm((self.r - host.r) - sep_length / 2 * self.p, axis=1)
        )

    def update(self,
               host,
               theta,
               theta_std,
               r_std=None,
               v_std=None,
               a_std=None):

        deviation = np.stack([
            (np.remainder(self.theta - theta + np.pi, 2 * np.pi) - np.pi) / theta_std,
        ], axis=1)
        logit = - np.sum(np.square(deviation), axis=1) / 2

        condition = np.stack([
            (np.linalg.norm(self.r - host.r, axis=1) / r_std) < 1,
        ], axis=1)
        filter_ = np.prod(condition, axis=1)

        self.likelihood = np.exp(logit - np.max(logit)) * filter_
        self.likelihood /= np.sum(self.likelihood)

    def measure(self,
                r_std=None,
                v_std=None,
                a_std=None):

        return Particles(
            num_particles=self.num_particles,
            r=self.r,
            r_std=r_std,
            v=self.v,
            v_std=v_std,
            a=self.a,
            a_std=a_std,
        )

    def estimate(self):
        if self.a is None:
            a = None
        else:
            a = np.dot(self.likelihood, self.a)

        if self.v is None:
            v = None
        else:
            v = np.dot(self.likelihood, self.v)

        if self.r is None:
            r = None
        else:
            r = np.dot(self.likelihood, self.r)

        return Particles(
            r=r,
            v=v,
            a=a,
        )

    def resample(self):
        num_particles_eff = 1. / np.sum(np.square(self.likelihood))
        print('Effective number of particles={:.2f}'.format(num_particles_eff))

        if num_particles_eff < self.num_particles * self.resample_ratio:
            indices = np.random.choice(
                self.num_particles,
                size=self.num_particles,
                p=self.likelihood,
            )

            if self.a is not None:
                self.a = self.a[indices]
            if self.v is not None:
                self.v = self.v[indices]
            if self.r is not None:
                self.r = self.r[indices]


host = Particles(
    r=(0, 0),
)

particles = Particles(
    num_particles=10000,
    r=host.r,
    r_std=3.0,
    v=(0, 0),
    v_std=0.5,
)

#

num_bins = 512
power = 0.3

fig = plot.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1)
ax.plot(host.r[0, 0], host.r[0, 1], 'ro')

(est_line,) = ax.plot(0, 0, 'bo')

xedges = np.linspace(-map_width, map_width, num_bins)
yedges = np.linspace(-map_height, map_height, num_bins)
(hist, _, _) = np.histogram2d(particles.r[:, 0], particles.r[:, 1], bins=[xedges, yedges])

img = ax.imshow(np.power(hist.T, power), extent=[-map_width, map_width, -map_height, map_height], origin='lower', cmap='binary')

ax.set_xlim(-map_width, map_width)
ax.set_ylim(-map_height, map_height)

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
        v=(0.2, 0.0),
        v_std=0.05,
    )
    particles.update(
        host=host,
        theta=phase[num_sample],
        theta_std=0.1,
        r_std=5.0,
    )
    particles.resample()
    est_client = particles.estimate()

    est_line.set_data([est_client.r[0, 0]], [est_client.r[0, 1]])
    (hist, _, _) = np.histogram2d(particles.r[:, 0], particles.r[:, 1], bins=[xedges, yedges])
    img.set_data(np.power(hist.T, power))
    img.autoscale()

    phase_line.set_data(np.arange(num_sample) * delta_t, phase[:num_sample])

    fig.canvas.draw()
    fig.canvas.flush_events()

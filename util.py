import numpy as np


class World(object):
    SpeedOfLight = 2.99792458e8

    def __init__(self,
                 delta_t,
                 wave_freq,
                 sep_length):

        self.delta_t = delta_t
        self.wave_number = (2 * np.pi * wave_freq) / World.SpeedOfLight
        self.sep_length = sep_length


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
                 world,
                 num_particles=1,
                 resample_ratio=1.0,
                 communism_ratio=0.25,
                 r=None,
                 r_std=None,
                 v=None,
                 v_std=None,
                 a=None,
                 a_std=None):

        self.world = world
        self.num_particles = num_particles
        self.resample_ratio = resample_ratio
        self.communism_ratio = communism_ratio

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

        self.dr = np.array([1, 0])  # TODO

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
            v = self.v + self.world.delta_t * self.a

        if v is not None:
            self.v = self.normal(v, v_std)
            r = self.r + self.world.delta_t * self.v

        if r is not None:
            self.r = self.normal(r, r_std)

        # measurement: no noise
        # phase.shape = (num_particles, num_subcarriers)
        self.phase = np.outer((
                np.linalg.norm((self.r - host.r) + self.world.sep_length / 2 * self.dr, axis=1) -
                np.linalg.norm((self.r - host.r) - self.world.sep_length / 2 * self.dr, axis=1)
            ),
            self.world.wave_number,
        )

    def update(self,
               host,
               phase,
               phase_spread,
               r_spread):

        # phase.shape = (num_subcarriers,)
        deviation = np.cos(self.phase - phase) / np.square(phase_spread)
        logit = np.sum(deviation, axis=1)

        condition = np.stack([
            (np.linalg.norm(self.r - host.r, axis=1) / r_spread) < 1,
        ], axis=1)
        filter_ = np.prod(condition, axis=1)

        self.likelihood = np.exp(logit - np.max(logit)) * filter_
        self.likelihood /= np.sum(self.likelihood)

        self.likelihood = (
            self.likelihood * (1 - self.communism_ratio) +
            np.ones(self.num_particles) / self.num_particles * self.communism_ratio
        ) * filter_
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
            world=self.world,
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

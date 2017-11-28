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

    def uniform(self):
        return np.ones((self.num_particles, 1)) / self.num_particles

    def smooth(self, prob):
        return (prob * (1 - self.smooth_ratio) + self.uniform() * self.smooth_ratio)

    def normalize(self, prob, mask=None):
        if mask is not None:
            prob = prob * mask
        prob = prob / np.sum(prob, axis=0, keepdims=True)
        return prob

    def softmax(self, logit, mask=None):
        prob = np.exp(logit - np.max(logit, axis=0, keepdims=True))
        prob = self.normalize(prob, mask=mask)
        return prob

    def entropy(self, prob, epsilon=1e-9):
        prob = self.normalize(prob + epsilon)
        entropy = - np.sum(prob * np.log(prob), axis=0)
        return entropy

    def __init__(self,
                 world,
                 num_particles=1,
                 resample_ratio=1.0,
                 smooth_ratio=0.25,
                 r=None,
                 r_std=None,
                 v=None,
                 v_std=None,
                 a=None,
                 a_std=None):

        self.world = world
        self.num_particles = num_particles
        self.resample_ratio = resample_ratio
        self.smooth_ratio = smooth_ratio

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
        self.likelihood = np.ones((num_particles, 1)) / num_particles

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

        logit = np.cos(self.phase - phase) / np.square(phase_spread)
        mask = (np.linalg.norm(self.r - host.r, axis=1, keepdims=True) / r_spread) < 1

        # '''
        likelihood = self.softmax(logit, mask=mask * self.likelihood)
        likelihood = self.normalize(self.smooth(likelihood) * mask)
        select = self.entropy(self.likelihood) < self.entropy(likelihood)

        if np.any(select):
            logit = logit[:, select]
            print('Eliminated {:d} features'.format(np.sum(np.logical_not(select))))
        # '''

        logit = np.sum(logit, axis=1, keepdims=True)

        self.likelihood = self.softmax(logit, mask=mask)
        self.likelihood = self.normalize(self.smooth(self.likelihood) * mask)

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
                p=self.likelihood[:, 0],
            )

            if self.a is not None:
                self.a = self.a[indices]
            if self.v is not None:
                self.v = self.v[indices]
            if self.r is not None:
                self.r = self.r[indices]

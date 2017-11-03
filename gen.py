import numpy as np

from util import World, Particles

t = 20
delta_t = 0.1
num_samples = int(t / delta_t)

world = World(
    delta_t=delta_t,
    wave_freq=900e6,
    sep_length=0.19,
)

host = Particles(
    world=world,
    r=(0, 0),
)

client = Particles(
    world=world,
    r=(-3.0, -3.5),
)

r = []
phase = []
for num_sample in range(num_samples):
    client.predict(
        host=host,
        v=(+0.2, +0.0),
    )

    r.append(client.r[0])
    phase.append(client.normal(client.phase, std=0.1)[0, 0])

np.savez(
    'data/gen.npz',
    delta_t=delta_t,
    num_samples=num_samples,
    r=r,
    phase=phase,
)

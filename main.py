import matplotlib
matplotlib.use('qt4agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', size=16)

import asyncore
import matplotlib.pyplot as plot
import numpy as np
import scipy.ndimage
import signal
import socket
import threading
import time

from phy import World, Particles


class Handler(asyncore.dispatcher):
    def __init__(self, sock, packet_size, read_callback):
        asyncore.dispatcher.__init__(self, sock)

        self.packet_size = packet_size
        self.read_callback = read_callback

    def handle_read(self):
        self.string = self.recv(self.packet_size)
        self.read_callback()


class Device(object):
    def __str__(self):
        return 'name={:s}, count={:3d}, len(value)={:d}'.format(self.name, self.count, len(self.value))

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

        self.lock = threading.Lock()
        self.handler = None

        self.is_valid = False
        self.count = None
        self.value = None

    def handle(self, sock, packet_size):
        self.handler = Handler(sock, packet_size=packet_size, read_callback=self.read)

    def read(self):
        string = self.handler.string

        if string and (string[0] == '$'):
            is_valid = True
            count = ord(string[2])
            value = np.frombuffer(string[3:-2], dtype=self.dtype)

            with self.lock:
                self.is_valid = is_valid
                self.count = count
                self.value = value


class Server(asyncore.dispatcher):
    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)

        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            (sock, addr) = pair
            print('Incoming connection from {:s}'.format(addr))

            data = sock.recv(5)
            Manager.Devices[ord(data[1])].handle(sock, packet_size=ord(data[2]))

    def start(self):
        self.thread = threading.Thread(target=asyncore.loop)
        self.thread.daemon = True
        self.thread.start()


class Manager(object):
    class ID:
        Arduino = 0
        ArduinoDebug = 1
        UE = 128

    Devices = {
        ID.Arduino: Device('Arduino', dtype=np.dtype('int16').newbyteorder('<')),
        ID.ArduinoDebug: Device('ArduinoDebug', dtype=np.dtype('int16').newbyteorder('<')),
        ID.UE: Device('UE', dtype=np.dtype('float32')),
    }

    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.step = 0
        self.phases = []

    def _run(self):
        valid = True
        values = {}
        strings = []
        for (id, device) in Manager.Devices.items():
            with device.lock:
                if device.is_valid:
                    values[id] = device.value.copy()
                    strings.append(str(device))
                else:
                    valid = False

        if strings:
            print('[{:.4f}] {:s}'.format(
                time.time(),
                ' | '.join(strings),
            ))
        else:
            print('.')

        if valid:
            a = (values[Manager.ID.Arduino][:2] / 16384) * 9.81
            phase = values[Manager.ID.UE]

            particles.predict(
                host=host,
                r_std=None,
                a=a,
                a_std=0.001,
            )
            particles.update(
                host=host,
                phase=phase,
                phase_spread=1.0,
                r_spread=size,
            )
            particles.resample()

            (hist, _, _) = np.histogram2d(particles.r[:, 0], particles.r[:, 1], bins=[xedges, yedges])
            (xs, ys) = np.where(np.logical_and.reduce([
                hist != 0,
                hist == scipy.ndimage.filters.maximum_filter(hist, size=local_bins),
            ]))

            est_line.set_offsets(np.stack([xedges[xs], yedges[ys]], axis=1))
            est_line.set_sizes(hist[xs, ys] / np.max(hist) * 30)
            img.set_data(np.power(hist.T / np.max(hist), power))
            img.autoscale()

            phase_line.set_data(np.arange(self.step) * delta_t, self.phases)
            x_max_step = self.step
            x_min_step = max(0, x_max_step - 500)
            phase_fig.axes[0].set_xlim(x_min_step * self.delta_t, x_max_step * self.delta_t)

            for fig in [map_fig, phase_fig]:
                fig.canvas.draw()
                fig.canvas.flush_events()

            self.step += 1
            self.phases.append(phase[0])

    def run(self):
        at = time.time()
        while True:
            self._run()

            at += self.delta_t
            sleep = at - time.time()
            if (sleep < 0):
                print('Overtime by {:.3f} seconds'.format(-sleep))
                at = time.time()
            else:
                time.sleep(sleep)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

################################################################################


if __name__ == '__main__':
    num_subcarriers = 50 * 12
    step_subcarriers = 12

    delta_t = 0.1
    wave_freq = 915e6 + 15e3 * (np.arange(0, num_subcarriers, step_subcarriers) - num_subcarriers / 2)
    sep_length = 0.19
    size = 20

    world = World(
        delta_t=delta_t,
        wave_freq=wave_freq,
        sep_length=sep_length,
    )

    host = Particles(
        world=world,
        r=(0.0, 0.0),
    )

    particles = Particles(
        world=world,
        num_particles=1000,
        smooth_ratio=0.25,
        r=host.r,
        r_std=size,
        v=(0.0, 0.0),
        v_std=0.05,
    )

    map_size = size
    num_bins = 512
    power = 0.3
    local_size = 3.0
    local_bins = int(local_size / map_size * num_bins)
    xedges = np.linspace(-map_size, map_size, num_bins)
    yedges = np.linspace(-map_size, map_size, num_bins)

    #

    map_fig = plot.figure(figsize=(6, 6))

    ax = map_fig.add_axes([0.15, 0.1, 0.8, 0.8])
    (host_line,) = ax.plot(host.r[0, 0], host.r[0, 1], 'ks')

    est_line = ax.scatter([], [], c='k')
    ax.legend(
        [host_line, est_line],
        ['TX', 'RX modes'],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.125),
        ncol=2,
    )
    (hist, _, _) = np.histogram2d(particles.r[:, 0], particles.r[:, 1], bins=[xedges, yedges])
    img = ax.imshow(
        np.power(hist.T / np.max(hist), power),
        extent=np.array([-map_size, map_size, -map_size, map_size]) - map_size / num_bins,
        origin='lower',
        cmap='binary',
    )

    ax.set_xlim(-map_size, map_size)
    ax.set_ylim(-map_size, map_size)

    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')

    plot.show(block=False)
    map_fig.canvas.draw()
    map_fig.canvas.flush_events()

    #

    phase_fig = plot.figure(figsize=(6, 6))

    ax = phase_fig.add_axes([0.15, 0.1, 0.8, 0.8])
    (phase_line,) = ax.plot([], [], 'k')
    ax.legend(
        [phase_line],
        ['915 MHz'],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.125),
    )

    ax.grid(True, linestyle=':')
    ax.set_ylim(-np.pi, np.pi)

    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$\phi$ (radian)')

    plot.show(block=False)
    phase_fig.canvas.draw()
    phase_fig.canvas.flush_events()

    #

    server = Server('0.0.0.0', 6006)
    server.start()

    manager = Manager(delta_t=0.1)
    manager.run()

    signal.pause()

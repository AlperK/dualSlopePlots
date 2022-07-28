import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def linearize_amplitudes(arr, sep):
    return np.log(np.square(sep) * arr)


def get_slopes(arr, sep):
    return np.diff(arr) / np.diff(sep)


def get_optical_properties(s_ac, s_p, f):
    w = 2 * np.pi * f
    n = 1.4
    c = 2.998e11

    absorption_coefficient = (w / (2 * (c/n))) * (np.divide(s_p, s_ac) - np.divide(s_ac, s_p))

    scattering_coefficient = (np.square(s_ac) - np.square(s_p)) / (3 * absorption_coefficient)

    return absorption_coefficient, scattering_coefficient


def plot_raw_amplitude_phase(ac, phi):
    fig, axes = plt.subplots(2, 2)
    ax = axes[0][0]
    ax.plot(ac.T[0][0][1])

    ax = axes[0][1]
    ax.plot(phi.T[0][0][1])

    ax = axes[1][0]
    ax.plot(ac.T[1][0][1])

    ax = axes[1][1]
    ax.plot(phi.T[1][0][1])

    fig.tight_layout()


date = Path('2022-07-28')
measurement = Path('DUAL-SLOPE-690')
measurementCount = Path('3')
location = Path.joinpath(date, measurement, measurementCount)

amplitudeLocation = Path.joinpath(location, Path('amplitude.csv'))
phaseLocation = Path.joinpath(location, Path('phase.csv'))

amplitudes = np.loadtxt(str(amplitudeLocation), delimiter=',')
amplitudes = amplitudes.reshape((amplitudes.shape[0], 2, 2, 2))

phases = np.loadtxt(str(phaseLocation), delimiter=',')
phases = phases.reshape((phases.shape[0], 2, 2, 2))

separations = np.array(([
    [[25, 35], [35, 25]],
    [[25, 35], [35, 25]]
]))

with open(Path.joinpath(location, Path('measurement settings.json')), 'r') as f:
    settings = json.load(f)

frequency = float(settings['RF']) * 1e6 + float(settings['IF']) * 1e3

linearizedAmplitudes = linearize_amplitudes(amplitudes, separations)

plot_raw_amplitude_phase(ac=amplitudes, phi=phases)

amplitude_slopes = get_slopes(linearizedAmplitudes, separations)
# amplitude_slopes[0] -> 820-1
# amplitude_slopes[1] -> 690-1
# amplitude_slopes[2] -> 820-2
# amplitude_slopes[3] -> 690-2

phase_slopes = get_slopes(np.deg2rad(phases), separations)
# phase_slopes[0][0][0] -> 820-1
# phase_slopes[0][0][1] -> 690-1
# phase_slopes[0][1][0] -> 820-2
# phase_slopes[0][1][1] -> 690-2


mu_a, mu_s = get_optical_properties((amplitude_slopes.T[0][0][1] + amplitude_slopes.T[0][1][1]) / 2,
                                    (phase_slopes.T[0][0][1] + phase_slopes.T[0][1][1]) / 2,
                                    frequency)

fig, axes = plt.subplots(2)
ax = axes[0]
ax.plot(mu_a)

ax = axes[1]
ax.plot(mu_s)

# print(linearizedAmplitudes[0])
# print('asd')
# print('asd')
# print('asd')
# print(phase_slopes[0])
# print('asd')
# print('asd')
# print('asd')
# print(phase_slopes.T[0][0][1])

plt.show()

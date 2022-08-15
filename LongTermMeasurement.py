import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def rolling_apply(fun, a, w):
    r = np.empty(a.shape)
    r.fill(np.nan)

    for i in range(w - 1, a.shape[0]):
        if fun is not None:
            r[i] = fun(a[(i-w+1):i+1])
        else:
            r[i] = a[(i - w + 1):i + 1]
    return r


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


def plot_raw_amplitude_phase(ac, ph, window=None):
    if window is not None:
        ac_1 = rolling_apply(fun=np.mean, a=ac.T[0][0][1], w=window)
        ac_2 = rolling_apply(fun=np.mean, a=ac.T[1][0][1], w=window)
        ph_1 = rolling_apply(fun=np.mean, a=ph.T[0][0][1], w=window)
        ph_2 = rolling_apply(fun=np.mean, a=ph.T[1][0][1], w=window)
    else:
        ac_1 = ac.T[0][0][1]
        ac_2 = ac.T[1][0][1]
        ph_1 = ph.T[0][0][1]
        ph_2 = ph.T[1][0][1]

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle(r'Raw AC and $\phi$ for Pair 1')

    ax = axes[0][0]
    ax.plot(ac_1,
            linewidth=2, color='brown')
    ax.set_title('Pair 1 AC at 25 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel('AC (mV)')

    ax = axes[0][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_1, w=window) / rolling_apply(fun=np.mean, a=ac_1, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 1 at 25 mm AC std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel('%')

    ax = axes[0][2]
    ax.plot(ph_1,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 1 $\phi$ at 25 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[0][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_1, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 1 at 25 mm $\phi$ std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][0]
    ax.plot(ac_2,
            linewidth=2, color='brown')
    ax.set_title('Pair 1 AC at 35 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel('AC (mV)')

    ax = axes[1][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_2, w=window) / rolling_apply(fun=np.mean, a=ac_2, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 1 AC at 35 mm std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel('%')

    ax = axes[1][2]
    ax.plot(ph_2,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 1 $\phi$ at 35 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_2, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 1 at 35 mm $\phi$ std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    fig.tight_layout()


def plot_optical_parameters(absorption, scattering, window=None):
    if window is not None:
        absorption = rolling_apply(fun=np.mean, a=absorption, w=window)
        mean_absorption = rolling_apply(fun=np.mean, a=absorption, w=window)
        std_absorption = rolling_apply(fun=np.std, a=absorption, w=window)

        scattering = rolling_apply(fun=np.mean, a=scattering, w=window)
        mean_scattering = rolling_apply(fun=np.mean, a=scattering, w=window)
        std_scattering = rolling_apply(fun=np.std, a=scattering, w=window)

    else:
        mean_absorption = np.mean(absorption)
        std_absorption = np.std(scattering)

        mean_scattering = np.mean(scattering)
        std_scattering = np.std(scattering)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    ax = axes[0][0]
    ax.scatter(np.arange(absorption.size), absorption,
               alpha=0.6, linewidth=2, color='brown', edgecolor='darkred')
    ax.axhline(0.0081, color='black', alpha=1, linewidth=3, linestyle=':')
    # ax.set_ylim([0.006, 0.012])
    ax.set_title(f'Absorption Coefficient')
    ax.set_xlabel(f'Data points')
    ax.set_ylabel(f'1/mm')

    ax = axes[0][1]
    ax.plot(100 * std_absorption / mean_absorption,
            color='brown', linewidth=2)
    ax.axhline(1, color='darkred', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('std / mean')
    ax.set_ylabel('%')
    ax.set_xlabel(f'Data points')

    ax = axes[1][0]
    ax.scatter(np.arange(scattering.size), scattering,
               alpha=0.6, linewidth=2, color='darkslateblue', edgecolor='darkblue')
    ax.axhline(0.761, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(f'Reduced Scattering Coefficient')
    ax.set_xlabel(f'Data points')
    ax.set_ylabel(f'1/mm')

    ax = axes[1][1]
    ax.plot(100 * std_scattering / mean_scattering,
            color='darkslateblue', linewidth=2)
    ax.axhline(1, color='darkblue', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('std / mean')
    ax.set_ylabel('%')
    ax.set_xlabel(f'Data points')

    fig.tight_layout()


date = Path('2022-07-29')
measurement = Path('DUAL-SLOPE-690')
measurementCount = Path('2')
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
window = 10
linearizedAmplitudes = linearize_amplitudes(amplitudes, separations)

plot_raw_amplitude_phase(ac=amplitudes, ph=phases, window=window)

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

plot_optical_parameters(mu_a, mu_s, window=window)

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

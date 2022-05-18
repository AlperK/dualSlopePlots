# TODO
# Add a the masking ability to delete stragglers
# Add a ability to pass plot names
# Add a new function for dual slope style plots
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv
from pathlib import Path as Path


def voltage2phase(phase_voltage, p):
    p[-1] = p[-1] - phase_voltage
    roots = np.roots(p)
    for root in roots:
        if (np.isreal(root)) and (root.real > 0) and (root.real < 180):
            return root


def voltage2phase4sine(phase_voltage, p):
    x = (phase_voltage - p[3]) / (p[0] * p[1])
    y = np.arcsin(x)
    phase = (y / (2 * np.pi * p[2])) - 90
    return -phase


def rolling_apply(fun, a, w):
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i-w+1):i+1])
    return r


def plot_amplitude_nsr(amp, wavelength, windowed=True, window_size=5, title=None):
    wavelength = str(wavelength)
    if wavelength == '690':
        color = 'darkslateblue'
        edgecolor = 'darkblue'
    elif wavelength == '820':
        color = 'brown'
        edgecolor = 'darkred'
    else:
        print('Wrong wavelength!')

    fig, axes = plt.subplots(4, figsize=(8, 8), num='{} {} nm Amplitude'.format(title, wavelength))
    fig.suptitle('{} {} nm Zoomed Amplitude'.format(title, wavelength))
    if windowed:
        mean = rolling_apply(np.mean, amp, window_size)
        std = rolling_apply(np.std, amp, window_size)
    else:
        mean = np.mean(amp)
        std = np.std(amp)

    ax = axes[0]
    ax.scatter(np.arange(amp.size), 1000 * amp,
               color=color, alpha=0.6, linewidth=2, edgecolor=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Measurement #')
    ax.set_ylabel('Voltage(mV)')
    ax.set_title('Raw amplitude')

    ax = axes[1]
    ax.hist(1000 * amp, bins='fd', color=color, alpha=0.6,
            linewidth=2, edgecolor=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Count')
    ax.set_title('Raw amplitude')

    ax = axes[2]
    ax.plot(100 * std / mean, color=color, alpha=0.6,
            linewidth=2, label='{} nm'.format(wavelength))
    ax.axhline(0.2, color=color, alpha=1, linewidth=3, linestyle=':')
    ax.set_xlabel('Measurement #')
    ax.set_ylabel('Percent')
    ax.set_title('Std / Mean, window size = {} '.format(window_size))

    ax = axes[3]
    ax.hist(100 * std / mean, color=color, alpha=0.6, bins='fd',
            linewidth=2, edgecolor=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Noise / Signal (%)')
    ax.set_ylabel('Count')
    ax.set_title('Window size = {}'.format(windowSize))

    fig.tight_layout()
    fig.savefig(Path.joinpath(saveLoc, Path('{} {} nm Amplitude'.format(title, wavelength))),
                bbox_inches='tight', dpi=800)


def plot_phase(phase, wavelength, windowed=True, window_size=5, title=None):
    wavelength = str(wavelength)
    if wavelength == '690':
        color = 'darkslateblue'
        edgecolor = 'darkblue'
    elif wavelength == '820':
        color = 'brown'
        edgecolor = 'darkred'
    else:
        print('Wrong wavelength!')

    fig, axes = plt.subplots(4, figsize=(8, 8), num='{} {} nm Phase'.format(title, wavelength))
    fig.suptitle('{} {} nm Zoomed Phase'.format(title, wavelength))
    if windowed:
        mean = rolling_apply(np.mean, phase, window_size)
        std = rolling_apply(np.std, phase, window_size)
    else:
        mean = np.mean(phase)
        std = np.std(phase)

    ax = axes[0]
    ax.scatter(np.arange(phase.size), phase, color=color, alpha=0.6,
               linewidth=2, edgecolors=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Measurement #')
    ax.set_ylabel('Degrees(°)')
    ax.set_title('Raw phase')

    ax = axes[1]
    ax.hist(phase, color=color, alpha=0.6, bins='fd',
            linewidth=2, edgecolor=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Degrees(°)')
    ax.set_ylabel('Count')
    ax.set_title('Raw phase')

    ax = axes[2]
    ax.plot(std, color=color, alpha=0.6,
            linewidth=2, label='{} nm'.format(wavelength))
    ax.axhline(0.2, color=color, alpha=1.0,
               linewidth=3, linestyle=':')
    ax.set_xlabel('Measurement #')
    ax.set_ylabel('Degrees(°)')
    ax.set_title('Std, window size = {} '.format(window_size))

    ax = axes[3]
    ax.hist(std, bins='fd', color=color, alpha=0.6,
            linewidth=2, edgecolor=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Degrees(°)')
    ax.set_ylabel('Count')
    ax.set_title('Raw phase')

    fig.tight_layout()
    fig.savefig(Path.joinpath(saveLoc, Path('{} {} nm Phase'.format(title, wavelength))), bbox_inches='tight', dpi=800)


def read_amplitudes_from_csv(save_location, demodulator_coefficients):
    amplitudes = np.loadtxt(Path.joinpath(save_location, Path('amplitude.csv')),
                            skiprows=1, delimiter=',')
    amplitudes = amplitudes / demodulator_coefficients['Amplitude Slope']
    return amplitudes


def read_phases_from_csv(save_location, amplitudes, demodulator_coefficients):
    phases = np.loadtxt(Path.joinpath(save_location, Path('phase.csv')),
                        skiprows=1, delimiter=',')
    for i, (amplitude, phase) in enumerate(zip(amplitudes, phases)):
        for j, (a, p) in enumerate(zip(amplitude, phase)):
            # phases[i][j] = voltage2phase(p, a * demodulator_coefficients['Phase Coefficients'])
            phases[i][j] = voltage2phase4sine(1000*p, [a, 303, 0.00277, -0.3])
            print(f'index {j}, Phase {phases[i][j]}, Amplitude {a}, PhaseV {p}')
    return phases


def linearize_intensity(amplitude, separation):
    return np.log(separation**2 * amplitude)


demodulator1Coefficients = {'Amplitude Slope': 0.3011,
                            'Phase Coefficients': np.array([1.6e-7, -4.3e-5, 2.6e-4, 0.2085])
                            }
demodulator2Coefficients = {'Amplitude Slope': 0.3011,
                            'Phase Coefficients': np.array([1.6e-7, -4.3e-5, 2.6e-4, 0.2085])
                            }
# saveLoc = Path.joinpath(Path('2022-05-16'), Path('DUAL-SLOPE-690'), Path('3'))
saveLoc = Path.joinpath(Path('2022-05-18'), Path('DUAL-SLOPE-820'), Path('5'))

mask = [39, 71, 138, 167, 222, 254, 268]
windowSize = 5


amplitudes = read_amplitudes_from_csv(saveLoc,
                                      demodulator_coefficients=demodulator1Coefficients)
phases = read_phases_from_csv(saveLoc, amplitudes=amplitudes,
                              demodulator_coefficients=demodulator1Coefficients)

plot_amplitude_nsr(amplitudes.T[0], 690, window_size=windowSize, title='Laser 1 APD 1')
plot_phase(phases.T[0], 690, window_size=windowSize, title='Laser 1 APD 1')
plot_amplitude_nsr(amplitudes.T[1], 690, window_size=windowSize, title='Laser 1 APD 2')
plot_phase(phases.T[1], 690, window_size=windowSize, title='Laser 1 APD 2')
#
plot_amplitude_nsr(amplitudes.T[2], 690, window_size=windowSize, title='Laser 2 APD 1')
plot_phase(phases.T[2], 690, window_size=windowSize, title='Laser 2 APD 1')
plot_amplitude_nsr(amplitudes.T[3], 690, window_size=windowSize, title='Laser 2 APD 2')
plot_phase(phases.T[3], 690, window_size=windowSize, title='Laser 2 APD 2')

nans = np.argwhere(np.isnan(phases))
print(nans)

nans = nans.ravel()[::2]
noNanPhase = np.delete(phases.T[7], nans)
nans = np.argwhere(np.isnan(noNanPhase))
print(nans.ravel()[::2])


# plot_amplitude_nsr(amplitudes.T[4] / amplitudes.T[5], 690, window_size=windowSize, title='Laser 1 APD 1 APD 2')
# plot_amplitude_nsr(amplitudes.T[7] / amplitudes.T[6], 690, window_size=windowSize, title='Laser 2 APD 1 APD 2')
# plot_amplitude_nsr((amplitudes.T[4] / amplitudes.T[5]) / (amplitudes.T[7] / amplitudes.T[6]), 690,
#                    window_size=windowSize, title='Laser 1 2 APD 1 APD 2')
# plot_phase(phases.T[4] - phases.T[5], 690, window_size=windowSize, title='Laser 1 APD 1 APD 2')
# plot_phase(phases.T[7] - phases.T[6], 690, window_size=windowSize, title='Laser 2 APD 1 APD 2')
# plot_phase((phases.T[4] - phases.T[5]) - (phases.T[7] - phases.T[6]), 690,
#            window_size=windowSize, title='Laser 1 2 APD 1 APD 2')

separations = np.array([25, 35])
linearizedIntensities = np.array([linearize_intensity(np.mean(amplitudes.T[3]), separations[0]),
                                  linearize_intensity(np.mean(amplitudes.T[2]), separations[1])
                                  ])
normalizedIntensities = linearizedIntensities - linearizedIntensities[0]
normalizedPhases = np.array([np.mean(phases.T[3]), np.mean(phases.T[2])])
normalizedPhases -= normalizedPhases[0]

intensitySlope = (normalizedIntensities[0] - normalizedIntensities[1]) / (separations[0] - separations[1])
print(f'Intensity Slope = {intensitySlope} per mm.')

phaseSlope = (normalizedPhases[0] - normalizedPhases[1]) / (separations[0] - separations[1])
print(f'Phase Slope = {10 * phaseSlope} °/cm or {10 * np.deg2rad(phaseSlope)} rad/cm.')


fig, axes = plt.subplots(2, 1, figsize=(8, 8))
ax = axes[0]
ax.scatter(separations, normalizedIntensities)
ax.set_xlim([20, 40])
ax.set_xlabel(f'Source-Detector distance (mm)')
ax.set_ylabel(r'log($\rho^2*\iota$)')
# ax.set_ylabel(f'log(\u03C1)')
ax.set_title(f'Intensity Slope')
ax.grid(True)

ax = axes[1]
ax.scatter(separations, normalizedPhases)
ax.set_xlim([20, 40])
ax.set_xlabel(f'Source-Detector distance (mm)')
ax.set_ylabel(f'Normalized Phases (°)')
ax.set_title(f'Phase Slope')
ax.grid(True)
fig.tight_layout()
plt.show()

# corrcoeffPha = ma.corrcoef([ma.masked_invalid(phases.T[4]), ma.masked_invalid(phases.T[5]),
#                             ma.masked_invalid(phases.T[6]), ma.masked_invalid(phases.T[7])])

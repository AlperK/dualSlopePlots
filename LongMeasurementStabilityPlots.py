# TODO
# Convert the csv reading code to something with 2D array for easy indexing

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path as Path


def voltage2phase(phase_voltage, p):
    # print(phase_voltage)
    p[-1] = p[-1] - phase_voltage
    roots = np.roots(p)
    for root in roots:
        if (np.isreal(root)) and (root.real > 0) and (root.real < 180):
            return root


def rolling_apply(fun, a, w):
    r = np.empty(a.shape)
    r.fill(np.nan)
    for i in range(w - 1, a.shape[0]):
        r[i] = fun(a[(i-w+1):i+1])
    return r


def plot_amplitude_nsr(amp, wavelength, windowed=True, window_size=5):
    wavelength = str(wavelength)
    if wavelength == '690':
        color = 'darkslateblue'
        edgecolor = 'darkblue'
    elif wavelength == '820':
        color = 'brown'
        edgecolor = 'darkred'
    else:
        print('Wrong wavelength!')

    fig, axes = plt.subplots(4, figsize=(8, 8), num='{} nm Amplitude'.format(wavelength))
    fig.suptitle('{} nm Zoomed Amplitude'.format(wavelength))
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
    ax.hist(100 * std / mean, bins='fd', color=color, alpha=0.6,
            linewidth=2, edgecolor=edgecolor, label='{} nm'.format(wavelength))
    ax.set_xlabel('Noise / Signal (%)')
    ax.set_ylabel('Count')
    ax.set_title('Window size = {}'.format(windowSize))

    fig.tight_layout()
    fig.savefig(Path.joinpath(saveLoc, Path('{} nm Amplitude'.format(wavelength))),
                bbox_inches='tight', dpi=800)


def plot_phase(phase, wavelength, windowed=True, window_size=5):
    wavelength = str(wavelength)
    if wavelength == '690':
        color = 'darkslateblue'
        edgecolor = 'darkblue'
    elif wavelength == '820':
        color = 'brown'
        edgecolor = 'darkred'
    else:
        print('Wrong wavelength!')

    fig, axes = plt.subplots(4, figsize=(8, 8), num='{} nm Phase'.format(wavelength))
    fig.suptitle('{} nm Zoomed Phase'.format(wavelength))
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
    ax.hist(phase, bins='fd', color=color, alpha=0.6,
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
    fig.savefig(Path.joinpath(saveLoc, Path('{} nm Phase'.format(wavelength))), bbox_inches='tight', dpi=800)


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
            phases[i][j] = voltage2phase(p, a * demodulator_coefficients['Phase Coefficients'])
    return phases


demodulator_coefficients = {'Amplitude Slope': 0.2063,
                            'Phase Coefficients': np.array([1.6e-7, -4.3e-5, 2.6e-4, 0.2085])
                            }
saveLoc = Path.joinpath(Path('2022-04-26'), Path('TEST-FOUR-LEDs-APD1'), Path('1'))

amp = Path.joinpath(saveLoc, Path('amplitude.csv'))
amp3 = []
amp4 = []
amp7 = []
amp8 = []
with open(amp, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        amp7.append(float(row[1])/0.2063)
        amp8.append(float(row[3])/0.2063)
        amp3.append(float(row[5])/0.2063)
        amp4.append(float(row[7])/0.2063)

amp3 = np.array(amp3)
amp4 = np.array(amp4)
amp7 = np.array(amp7)
amp8 = np.array(amp8)


pha = Path.joinpath(saveLoc, Path('phase.csv'))
pha3 = []
pha4 = []
pha7 = []
pha8 = []
with open(pha, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        pha7.append(float(row[1]))
        pha8.append(float(row[3]))
        pha3.append(float(row[5]))
        pha4.append(float(row[7]))

phaseDegrees3 = np.zeros_like(pha3)
phaseDegrees4 = np.zeros_like(pha4)
phaseDegrees7 = np.zeros_like(pha7)
phaseDegrees8 = np.zeros_like(pha8)
for i, amp in enumerate(amp3):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees3[i] = voltage2phase(pha3[i], coefficients)

for i, amp in enumerate(amp4):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees4[i] = voltage2phase(pha4[i], coefficients)

for i, amp in enumerate(amp7):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees7[i] = voltage2phase(pha7[i], coefficients)

for i, amp in enumerate(amp8):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees8[i] = voltage2phase(pha8[i], coefficients)

mask = [39, 71, 138, 167, 222, 254, 268]
# mask = []

amp3 = np.delete(amp3, mask)
amp4 = np.delete(amp4, mask)
phaseDegrees3 = np.delete(phaseDegrees3, mask)
phaseDegrees4 = np.delete(phaseDegrees4, mask)

amp7 = np.delete(amp7, mask)
amp8 = np.delete(amp8, mask)
phaseDegrees7 = np.delete(phaseDegrees7, mask)
phaseDegrees8 = np.delete(phaseDegrees8, mask)

windowSize = 5


amplitudes = read_amplitudes_from_csv(saveLoc,
                                      demodulator_coefficients=demodulator_coefficients)
phases = read_phases_from_csv(saveLoc, amplitudes=amplitudes,
                     demodulator_coefficients=demodulator_coefficients)

plot_amplitude_nsr(amplitudes.T[3], 690, window_size=windowSize)
plot_phase(phases.T[3], 690, window_size=windowSize)
plot_amplitude_nsr(amplitudes.T[7], 820, window_size=windowSize)
plot_phase(phases.T[7], 820, window_size=windowSize)

plt.show()
# corrcoeffAmp = np.corrcoef(amp4, amp8)
# print(corrcoeffAmp)
# print()
# corrcoeffPha = np.corrcoef(phaseDegrees4, phaseDegrees8)
# print(corrcoeffPha)
# print()
# corrcoeffAmpPha690 = np.corrcoef(amp4, phaseDegrees4)
# print(corrcoeffAmpPha690)
# print()
# corrcoeffAmpPha830 = np.corrcoef(amp8, phaseDegrees8)
# print(corrcoeffAmpPha830)

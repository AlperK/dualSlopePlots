import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path as Path


def voltage2phase(voltage, p):
    p[-1] = p[-1]-voltage
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


saveLoc = Path.joinpath(Path('2022-04-19'), Path('TEST'), Path('1'))

amp = Path.joinpath(saveLoc, Path('amplitude.csv'))
amp4 = []
amp8 = []
with open(amp, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        amp8.append(float(row[3])/0.2063)
        amp4.append(float(row[7])/0.2063)

amp4 = np.array(amp4)
amp8 = np.array(amp8)


pha = Path.joinpath(saveLoc, Path('phase.csv'))
pha4 = []
pha8 = []
with open(pha, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        pha8.append(float(row[3]))
        pha4.append(float(row[7]))

phaseDegrees4 = np.zeros_like(pha4)
phaseDegrees8 = np.zeros_like(pha8)
for i, amp in enumerate(amp4):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees4[i] = voltage2phase(pha4[i], coefficients)

for i, amp in enumerate(amp8):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees8[i] = voltage2phase(pha8[i], coefficients)


windowSize = 5

#   690 nm Amplitude Plots
amp4_movingMean = rolling_apply(np.mean, amp4, windowSize)
amp4_movingStd = rolling_apply(np.std, amp4, windowSize)

fig, axes = plt.subplots(3, figsize=(8, 8), num='690 nm Amplitude')
fig.suptitle('690 nm Amplitude')

axes[0].scatter(np.arange(amp4.size), 1000 * amp4, color='darkslateblue', alpha=0.6,
                linewidth=2, edgecolors='darkblue', label='690nm')
axes[0].set_xlabel('Measurement #')
axes[0].set_ylabel('Voltage(mV)')
axes[0].set_title('Raw amplitude')

axes[1].plot(100 * amp4_movingStd / amp4_movingMean, color='darkslateblue', alpha=0.6,
             linewidth=2, label='690nm')
axes[1].set_xlabel('Measurement #')
axes[1].set_ylabel('Percent')
axes[1].set_title('Std / Mean, window size = {} '.format(windowSize))

axes[2].hist(1000 * amp4, bins='fd', color='darkslateblue', alpha=0.6,
             linewidth=2, edgecolor='darkblue', label='690 nm')
axes[2].set_xlabel('Voltage (mV)')
axes[2].set_ylabel('Count')
axes[2].set_title('Raw amplitude')

fig.tight_layout()
fig.savefig(Path.joinpath(saveLoc, Path('690 nm Amplitude')), bbox_inches='tight', dpi=800)


#   690 nm Phase Plots
pha4_movingMean = rolling_apply(np.mean, phaseDegrees4, windowSize)
pha4_movingStd = rolling_apply(np.std, phaseDegrees4, windowSize)

fig, axes = plt.subplots(3, figsize=(8, 8), num='690 nm Phase')
fig.suptitle('690 nm Phase')

axes[0].scatter(np.arange(phaseDegrees4.size), phaseDegrees4, color='darkslateblue', alpha=0.6,
                linewidth=2, edgecolors='darkblue', label='690nm')
axes[0].set_xlabel('Measurement #')
axes[0].set_ylabel('Degrees(°)')
axes[0].set_title('Raw phase')

axes[1].plot(pha4_movingStd, color='darkslateblue', alpha=0.6,
             linewidth=2, label='690nm')
axes[1].set_xlabel('Measurement #')
axes[1].set_ylabel('Degrees(°)')
axes[1].set_title('Std, window size = {} '.format(windowSize))

axes[2].hist(phaseDegrees4, bins='fd', color='darkslateblue', alpha=0.6,
             linewidth=2, edgecolor='darkblue', label='690 nm')
axes[2].set_xlabel('Degrees(°)')
axes[2].set_ylabel('Count')
axes[2].set_title('Raw phase')

fig.tight_layout()
fig.savefig(Path.joinpath(saveLoc, Path('690 nm Phase')), bbox_inches='tight', dpi=800)


#   830 nm Amplitude Plots
amp8_movingMean = rolling_apply(np.mean, amp8, windowSize)
amp8_movingStd = rolling_apply(np.std, amp8, windowSize)

fig, axes = plt.subplots(3, figsize=(8, 8), num='830 nm Amplitude')
fig.suptitle('830 nm Amplitude')

axes[0].scatter(np.arange(amp8.size), 1000 * amp8, color='brown', alpha=0.6,
                linewidth=2, edgecolors='darkred', label='830nm')
axes[0].set_xlabel('Measurement #')
axes[0].set_ylabel('Voltage(mV)')
axes[0].set_title('Raw amplitude')

axes[1].plot(100 * amp8_movingStd / amp8_movingMean, color='brown', alpha=0.6,
             linewidth=2, label='830nm')
axes[1].set_xlabel('Measurement #')
axes[1].set_ylabel('Percent')
axes[1].set_title('Std / Mean, window size = {} '.format(windowSize))

axes[2].hist(1000 * amp8, bins='fd', color='brown', alpha=0.6,
             linewidth=2, edgecolor='darkred', label='830 nm')
axes[2].set_xlabel('Voltage (mV)')
axes[2].set_ylabel('Count')
axes[2].set_title('Raw amplitude')

fig.tight_layout()
fig.savefig(Path.joinpath(saveLoc, Path('830 nm Amplitude')), bbox_inches='tight', dpi=800)


#   830 nm Phase Plots
pha8_movingMean = rolling_apply(np.mean, phaseDegrees8, windowSize)
pha8_movingStd = rolling_apply(np.std, phaseDegrees8, windowSize)

fig, axes = plt.subplots(3, figsize=(8, 8), num='830 nm Phase')
fig.suptitle('830 nm Phase')

axes[0].scatter(np.arange(phaseDegrees8.size), phaseDegrees8, color='brown', alpha=0.6,
                linewidth=2, edgecolors='darkred', label='830nm')
axes[0].set_xlabel('Measurement #')
axes[0].set_ylabel('Degrees(°)')
axes[0].set_title('Raw phase')

axes[1].plot(pha8_movingStd, color='brown', alpha=0.6,
             linewidth=2, label='830nm')
axes[1].set_xlabel('Measurement #')
axes[1].set_ylabel('Degrees(°)')
axes[1].set_title('Std, window size = {} '.format(windowSize))

axes[2].hist(phaseDegrees8, bins='fd', color='brown', alpha=0.6,
             linewidth=2, edgecolor='darkred', label='830 nm')
axes[2].set_xlabel('Degrees(°)')
axes[2].set_ylabel('Count')
axes[2].set_title('Raw phase')

fig.tight_layout()
fig.savefig(Path.joinpath(saveLoc, Path('830 nm Phase')), bbox_inches='tight', dpi=800)


#   Amplitude Ratio
ampRatio_movingMean = rolling_apply(np.mean, amp4 / amp8, windowSize)
ampRatio_movingStd = rolling_apply(np.std, amp4 / amp8, windowSize)

fig, axes = plt.subplots(3, figsize=(8, 8), num='Amplitude Ratio')
fig.suptitle('Amplitude Ratio')

axes[0].scatter(np.arange(amp8.size), amp4 / amp8, color='seagreen', alpha=0.6,
                linewidth=2, edgecolors='darkgreen', label='830nm')
axes[0].set_xlabel('Measurement #')
axes[0].set_ylabel('a.u.')
axes[0].set_title('Raw amplitude ratio')

axes[1].plot(100 * ampRatio_movingStd / ampRatio_movingMean, color='seagreen', alpha=0.6,
             linewidth=2, label='830nm')
axes[1].set_xlabel('Measurement #')
axes[1].set_ylabel('Percent')
axes[1].set_title('Std / Mean, window size = {} '.format(windowSize))

axes[2].hist(amp4 / amp8, bins='fd', color='seagreen', alpha=0.6,
             linewidth=2, edgecolor='darkgreen', label='830 nm')
axes[2].set_xlabel('a.u.')
axes[2].set_ylabel('Count')
axes[2].set_title('Histogram')

fig.tight_layout()
fig.savefig(Path.joinpath(saveLoc, Path('Amplitude Ratio')), bbox_inches='tight', dpi=800)


# Phase Differences
phaDiff_movingMean = rolling_apply(np.mean, phaseDegrees4 - phaseDegrees8, windowSize)
phaDiff_movingStd = rolling_apply(np.std, phaseDegrees4 - phaseDegrees8, windowSize)

fig, axes = plt.subplots(3, figsize=(8, 8), num='Phase Difference')
fig.suptitle('Phase Difference')

axes[0].scatter(np.arange(amp8.size), phaseDegrees4 - phaseDegrees8, color='seagreen', alpha=0.6,
                linewidth=2, edgecolors='darkgreen', label='830nm')
axes[0].set_xlabel('Measurement #')
axes[0].set_ylabel('Degrees (°)')
axes[0].set_title('Raw Phase Difference')

axes[1].plot(phaDiff_movingStd, color='seagreen', alpha=0.6,
             linewidth=2, label='830nm')
axes[1].set_xlabel('Measurement #')
axes[1].set_ylabel('Degrees (°)')
axes[1].set_title('Std, window size = {} '.format(windowSize))

axes[2].hist(phaseDegrees4 - phaseDegrees8, bins='fd', color='seagreen', alpha=0.6,
             linewidth=2, edgecolor='darkgreen', label='830 nm')
axes[2].set_xlabel('Degrees (°)')
axes[2].set_ylabel('Count')
axes[2].set_title('Histogram')

fig.tight_layout()
fig.savefig(Path.joinpath(saveLoc, Path('Phase Difference')), bbox_inches='tight', dpi=800)


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

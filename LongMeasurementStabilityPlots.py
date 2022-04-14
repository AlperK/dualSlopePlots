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


saveLoc = Path.joinpath(Path('2022-04-13'), Path('30-MIN-PHANTOM-3'), Path('2'))

amp = Path.joinpath(saveLoc, Path('amplitude.csv'))
amp4 = []
amp8 = []
with open(amp, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        amp4.append(float(row[3])/0.2063)
        amp8.append(float(row[7])/0.2063)

amp4 = np.array(amp4)
amp8 = np.array(amp8)


pha = Path.joinpath(saveLoc, Path('phase.csv'))
pha4 = []
pha8 = []
with open(pha, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        pha4.append(float(row[3]))
        pha8.append(float(row[7]))

phaseDegrees4 = np.zeros_like(pha4)
phaseDegrees8 = np.zeros_like(pha8)
for i, amp in enumerate(amp4):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees4[i] = voltage2phase(pha4[i], coefficients)

for i, amp in enumerate(amp8):
    coefficients = 1e-3 * amp * np.array([1.6e-4, -0.043, 0.26, 208.5])
    phaseDegrees8[i] = voltage2phase(pha8[i], coefficients)


#   690nm plots
plt.figure('690 nm Amplitude')
plt.scatter(np.arange(amp4.size), amp4, color='darkslateblue', alpha=0.6,
            linewidth=2, edgecolors='darkblue', label='690nm')
plt.figtext(0.2, 0.3, 'SNR = {}'.format(np.around(10*np.log10(np.mean(amp4)/np.std(amp4)), 2)))
plt.title('Raw measured amplitude')
plt.xlabel('Measurement #')
plt.ylabel('Voltage(V)')
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('690 nm Amplitude')), bbox_inches='tight', dpi=800)

plt.figure('690 nm Amplitude Hist')
plt.hist(amp4, bins='fd', color='darkslateblue', alpha=0.6,
         linewidth=2, edgecolor='darkblue', label='690 nm')
plt.title('Raw measured amplitude')
plt.xlabel('Voltage(V)')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('690 nm Amplitude Hist')), bbox_inches='tight', dpi=800)
print('Std for 690nm Amplitude: {}'.format(np.std(amp4)))
print('Mean for 690nm Amplitude: {}'.format(np.mean(amp4)))
print('10dB(mean/std) for 690nm Amplitude: {}'.format(10*np.log10(np.mean(amp4)/np.std(amp4))))

plt.figure('690 nm Phase')
plt.scatter(np.arange(phaseDegrees4.size), phaseDegrees4, color='darkslateblue', alpha=0.6,
            linewidth=2, edgecolor='darkblue', label='690 nm')
plt.figtext(0.2, 0.7, 'std = {}°'.format(np.around(np.std(phaseDegrees4), 2)))
plt.title('Raw measured phase')
plt.xlabel('Measurement #')
plt.ylabel('Degrees(°)')
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('690 nm Phase')), bbox_inches='tight', dpi=800)

plt.figure('690 nm Phase Hist')
plt.hist(phaseDegrees4, bins='fd', color='darkslateblue', alpha=0.6,
         linewidth=2, edgecolor='darkblue', label='690nm')
plt.title('Raw measured phase')
plt.xlabel('Degrees(°)')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
print('Std for 690nm Phase: {}'.format(np.std(phaseDegrees4)))
plt.savefig(Path.joinpath(saveLoc, Path('690 nm Phase Hist')), bbox_inches='tight', dpi=800)

#   830nm plots
plt.figure('830 nm Amplitude')
plt.scatter(np.arange(amp8.size), amp8, color='brown', alpha=0.6,
            linewidth=2, edgecolor='darkred', label='830nm')
plt.figtext(0.2, 0.3, 'SNR = {}'.format(np.around(10*np.log10(np.mean(amp8)/np.std(amp8)), 2)))
plt.title('Raw measured amplitude')
plt.xlabel('Measurement #')
plt.ylabel('Voltage(V)')
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('830 nm Amplitude')), bbox_inches='tight', dpi=800)

plt.figure('830 nm Amplitude Hist')
plt.hist(amp8, bins=30, color='brown', alpha=0.6,
         linewidth=2, edgecolor='darkred', label='830nm')
plt.title('Raw measured amplitude')
plt.xlabel('Voltage(V)')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
print('Std for 830nm Amplitude1: {}'.format(np.std(amp8)))
print('10dB(mean/std) for 830nm Amplitude: {}'.format(10*np.log10(np.mean(amp8)/np.std(amp8))))
plt.savefig(Path.joinpath(saveLoc, Path('830 nm Amplitude Hist')), bbox_inches='tight', dpi=800)


plt.figure('830 nm Phase')
plt.scatter(np.arange(phaseDegrees8.size), phaseDegrees8,
            color='brown', alpha=0.6, linewidth=2, edgecolor='darkred', label='830nm')
plt.figtext(0.2, 0.7, 'std = {}°'.format(np.around(np.std(phaseDegrees8), 2)))
plt.title('Raw measured phase')
plt.xlabel('Measurement #')
plt.ylabel('Degrees(°)')
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('830 nm Phase')), bbox_inches='tight', dpi=800)

plt.figure('830 nm Phase Hist')
plt.hist(phaseDegrees8, bins=30, label='830nm',
         color='brown', alpha=0.6, linewidth=2, edgecolor='darkred', )
plt.title('Raw measured phase')
plt.xlabel('Degrees(°)')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('830 nm Phase Hist')), bbox_inches='tight', dpi=800)
print('Std for 830nm Phase4: {}'.format(np.std(phaseDegrees8)))

# Amplitude Ratio
plt.figure('Amplitude Ratio')
plt.scatter(np.arange(amp8.size), amp4/amp8, label='690nm / 830nm',
            color='seagreen', alpha=0.6, linewidth=2, edgecolor='darkgreen')
plt.figtext(0.2, 0.3, 'SNR = {}'.format(np.around(10*np.log10(np.mean(amp4/amp8)/np.std(amp4/amp8)), 2)))
plt.title('Amplitude Ratio')
plt.xlabel('Measurement #')
plt.ylabel('a.u.')
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('Amplitude Ratio')), bbox_inches='tight', dpi=800)

plt.figure('Amplitude Ratio Hist')
plt.hist(amp4/amp8, bins='fd', label='690nm / 830nm',
         color='seagreen', alpha=0.6, linewidth=2, edgecolor='darkgreen')
plt.title('Amplitude Ratio')
plt.xlabel('Voltage(V)')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('Amplitude Ratio Hist')), bbox_inches='tight', dpi=800)
print('Std Amplitude Ratios: {}'.format(np.std(amp4/amp8)))
print('10dB(mean/std) for Amplitude Ratios: {}'.format(10*np.log10(np.mean(amp4/amp8)/np.std(amp4/amp8))))

# Phase Differences
fig = plt.figure('Phase Difference')
plt.scatter(np.arange(phaseDegrees4.size), phaseDegrees4 - phaseDegrees8, label='690nm - 830nm',
            color='seagreen', alpha=0.6, linewidth=2, edgecolor='darkgreen')
plt.figtext(0.2, 0.7, 'std = {}°'.format(np.around(np.std(phaseDegrees4 - phaseDegrees8), 2)))
plt.title('Phase Difference')
plt.xlabel('Measurement #', fontsize=14)
plt.ylabel('Degrees(°)', fontsize=14)
plt.tight_layout()
plt.legend()
plt.savefig(Path.joinpath(saveLoc, Path('Phase Differences')), bbox_inches='tight', dpi=800)

plt.figure('Phase Difference Hist')
plt.hist(phaseDegrees4 - phaseDegrees8, bins='fd', label='690nm - 830nm',
         color='seagreen', alpha=0.6, linewidth=2, edgecolor='darkgreen')
plt.title('Phase Difference')
plt.xlabel('Degrees(°)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.legend()
print('Std Phase Differences: {}'.format(np.std(phaseDegrees4 - phaseDegrees8)))
plt.savefig(Path.joinpath(saveLoc, Path('Phase Differences Hist')), bbox_inches='tight', dpi=800)
# plt.show()

corrCoeffAmp = np.corrcoef(amp4, amp8)
print(corrcoeffAmp)
print()
corrcoeffPha = np.corrcoef(phaseDegrees4, phaseDegrees8)
print(corrcoeffPha)
print()
corrcoeffAmpPha690 = np.corrcoef(amp4, phaseDegrees4)
print(corrcoeffAmpPha690)
print()
corrcoeffAmpPha830 = np.corrcoef(amp8, phaseDegrees8)
print(corrcoeffAmpPha830)

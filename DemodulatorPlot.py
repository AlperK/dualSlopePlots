import csv
from pathlib import Path as Path
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model


def third(x, amp, a, b, c, d, e):
    # amp = [0.25, 0.5, 0.75, 1.0]
    return amp*(a*x**3 + b*x**2 + c*x + d) + e


def line(x, a, b):
    return a*x + b


def sine_model(x, amp, a, freq, phi, offset):
    return amp * a * (np.sin(2*np.pi*freq*(x+90) + phi)) + np.log(amp)*offset


p0 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('PhaseCalibration'),
                   Path('1kHz_100mV_1000mV_phase.csv'))
pha0 = []
vol0 = []
with open(p0, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        pha0.append(int(row[0]))
        vol0.append(float(row[1])*1e3)

p1 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('PhaseCalibration'),
                   Path('1kHz_400mV_1000mV_phase.csv'))
pha1 = []
vol1 = []
with open(p1, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        pha1.append(int(row[0]))
        vol1.append(float(row[1])*1e3)

p2 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('PhaseCalibration'),
                   Path('1kHz_700mV_1000mV_phase.csv'))
pha2 = []
vol2 = []
with open(p2, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        pha2.append(int(row[0]))
        vol2.append(float(row[1])*1e3)

p3 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('PhaseCalibration'),
                   Path('1kHz_1000mV_1000mV_phase.csv'))
pha3 = []
vol3 = []
with open(p3, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        pha3.append(int(row[0]))
        vol3.append(float(row[1])*1e3)

a0 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('AmplitudeCalibration'), Path('1kHz_amplitude.csv'))
amp0 = []
a_vol0 = []
with open(a0, newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        amp0.append(int(row[0]))
        a_vol0.append(float(row[1])*1e3)

# a1 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('AmplitudeCalibration'), Path('4kHz_amplitude.csv'))
# amp1 = []
# a_vol1 = []
# with open(a1, newline='') as csv_file:
#     reader = csv.reader(csv_file, delimiter=',')
#     for row in reader:
#         amp1.append(int(row[0]))
#         a_vol1.append(float(row[1])*1e3)
#
# a2 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('AmplitudeCalibration'), Path('7kHz_amplitude.csv'))
# amp2 = []
# a_vol2 = []
# with open(a2, newline='') as csv_file:
#     reader = csv.reader(csv_file, delimiter=',')
#     for row in reader:
#         amp2.append(int(row[0]))
#         a_vol2.append(float(row[1])*1e3)
#
# a3 = Path.joinpath(Path('2022-06-08'), Path('Demodulator-2'), Path('AmplitudeCalibration'), Path('10kHz_amplitude.csv'))
# amp3 = []
# a_vol3 = []
# with open(a3, newline='') as csv_file:
#     reader = csv.reader(csv_file, delimiter=',')
#     for row in reader:
#         amp3.append(int(row[0]))
#         a_vol3.append(float(row[1])*1e3)


# PHASE FITS
tmodel = Model(third)

params = tmodel.make_params(a=3.32e-5, b=-0.009, c=0.065, d=0.211, amp=0.5, e=0)
params['amp'].vary = False
params['e'].vary = False
# params['a'].vary = False
# params['b'].vary = False
# params['c'].vary = False
result1 = tmodel.fit(vol0[45:135], params, x=pha0[45:135])

params = tmodel.make_params(a=1.555e-4, b=-0.0422, c=0.333, d=197.18, amp=1.0, e=0)
params['amp'].vary = False
params['e'].vary = False
# params['a'].vary = False
# params['b'].vary = False
# params['c'].vary = False
result2 = tmodel.fit(vol1[45:135], params, x=pha0[45:135])
# print(result2.fit_report())

params = tmodel.make_params(a=3.32e-5, b=-0.009, c=0.065, d=0.211, amp=1.5, e=0)
params['amp'].vary = False
params['e'].vary = False
params['a'].vary = False
params['b'].vary = False
params['c'].vary = False
result3 = tmodel.fit(vol2[45:135], params, x=pha0[45:135])
# print(result3.fit_report())

params = tmodel.make_params(a=3.32e-5, b=-0.009, c=0.065, d=0.211, amp=2.0, e=0)
params['amp'].vary = False
params['e'].vary = False
params['a'].vary = False
params['b'].vary = False
params['c'].vary = False
result4 = tmodel.fit(vol3[45:135], params, x=pha0[45:135])


# print(result1.fit_report())
# print(result2.fit_report())
# print(result3.fit_report())
# print(result4.fit_report())


# AMPLITUDE FITS
lmodel = Model(line)

params = lmodel.make_params(a=1, b=0)
# params['b'].vary = False
amplitudeFit = lmodel.fit(a_vol0, params, method='leastsq', x=amp0)
print(amplitudeFit.fit_report())
# result5 = lmodel.fit(a_vol0, params, x=amp0)
# result6 = lmodel.fit(a_vol1, params, x=amp1)
# result7 = lmodel.fit(a_vol2, params, x=amp2)
# result8 = lmodel.fit(a_vol3, params, x=amp3)
# print(result5.fit_report())
# print(result6.fit_report())
# print(result7.fit_report())
# print(result8.fit_report())

smodel = Model(sine_model)
params = smodel.make_params(amp=0.1, a=301.17, freq=2.7773e-3, phi=0, offset=0)
params['amp'].vary = False
# params['a'].vary = False
params['freq'].vary = False
params['phi'].vary = False
# params['offset'].vary = False
sine1 = smodel.fit(vol0[45:135], params, method='leastsq', x=pha0[45:135])

# params = smodel.make_params(amp=0.1, a=302.5, freq=2.77e-3, phi=0, offset=-1)
# params['amp'].vary = False
# params['a'].vary = False
# params['freq'].vary = False
# params['phi'].vary = False
# # params['offset'].vary = False
# sine1 = smodel.fit(vol0[45:135], params, method='leastsq', x=pha0[45:135])
#
# params = smodel.make_params(amp=0.2, a=302.5, freq=2.77e-3, phi=0, offset=-1)
# params['amp'].vary = False
# params['a'].vary = False
# params['freq'].vary = False
# params['phi'].vary = False
# # params['offset'].vary = False
# sine2 = smodel.fit(vol1[45:135], params, method='leastsq', x=pha0[45:135])
#
# params = smodel.make_params(amp=0.3, a=302.5, freq=2.77e-3, phi=0, offset=-1)
# params['amp'].vary = False
# params['a'].vary = False
# params['freq'].vary = False
# params['phi'].vary = False
# # params['offset'].vary = False
# sine3 = smodel.fit(vol2[45:135], params, method='leastsq', x=pha0[45:135])
#
# params = smodel.make_params(amp=0.4, a=301, freq=2.77e-3, phi=0, offset=1.6)
# params['amp'].vary = False
# params['a'].vary = False
# params['freq'].vary = False
# params['phi'].vary = False
# params['offset'].vary = False
# sine4 = smodel.fit(vol3[45:135], params, method='leastsq', x=pha0[45:135])

print(sine1.fit_report())
# print(sine2.fit_report())
# print(sine3.fit_report())
# print(sine4.fit_report())
#
# plt.figure(figsize=(6, 6))
plt.scatter(pha0, vol0, s=15, label=r'$V_{sig} = 100 mV$')
# plt.scatter(pha0, vol1, s=15, label=r'$V_{sig} = 200 mV$')
# plt.scatter(pha0, vol2, s=15, label=r'$V_{sig} = 300 mV$')
plt.scatter(pha0, vol3, s=15, label=r'$V_{sig} = 1000 mV$')
#
# plt.scatter(pha0, sine_model(np.array([pha0]), amp=0.1, a=303, freq=0.00277, phi=0, offset=-0.3), s=1, c='k',
#             label='SINE-FIT-0.1')
# plt.scatter(pha0, sine_model(np.array([pha0]), amp=0.2, a=303, freq=0.00277, phi=0, offset=-0.3), s=1, c='k',
#             label='SINE-FIT-0.2')
# plt.scatter(pha0, sine_model(np.array([pha0]), amp=0.3, a=303, freq=0.00277, phi=0, offset=-0.3), s=1, c='k',
#             label='SINE-FIT-0.3')
# plt.scatter(pha0, sine_model(np.array([pha0]), amp=0.4, a=303, freq=0.00277, phi=0, offset=-0.3), s=1, c='k',
#             label='SINE-FIT-0.4')
plt.scatter(pha0, sine_model(np.array([pha0]), amp=0.1, a=300.466, freq=0.00277773, phi=0, offset=-0.912), s=1, c='k',
            label='SINE-FIT-0.1V')
plt.scatter(pha0, sine_model(np.array([pha0]), amp=1.0, a=300.466, freq=0.00277773, phi=0, offset=-0.912), s=1, c='k',
            label='SINE-FIT-1.0V')
plt.legend()
plt.tight_layout()


# Phase Plot
# plt.figure(figsize=(6, 6))
# plt.scatter(pha0, vol0, s=15, label=r'$V_{sig} = 500 mV$')
# plt.scatter(pha1, vol1, s=15, label=r'$V_{sig} = 1000 mV$')
# plt.scatter(pha2, vol2, s=15, label=r'$V_{sig} = 1500 mV$')
# plt.scatter(pha3, vol3, s=15, label=r'$V_{sig} = 2000 mV$')
# plt.legend()

# plt.scatter(pha0, result1.best_fit, s=5, label='FIT')
# plt.scatter(pha0, third(np.array([pha0]), 0.5, 1.6e-4, -0.043, 0.26, 208.5, 0), s=1, c='k', label='FIT-0.5')
# plt.scatter(pha1, third(np.array([pha1]), 1.0, 1.6e-4, -0.043, 0.26, 208.5, 0), s=1, c='k', label='FIT-1.0')
# plt.scatter(pha2, third(np.array([pha2]), 1.5, 1.6e-4, -0.043, 0.26, 208.5, 0), s=1, c='k', label='FIT-1.5')
# plt.scatter(pha3, third(np.array([pha3]), 2.0, 1.6e-4, -0.043, 0.26, 208.5, 0), s=1, c='k', label='FIT-2.0')

# plt.scatter(pha3, third(np.array([pha3]), 0.5, 1.555e-4, -0.042, 0.333, 197.18, 0), s=1, c='k', label='NEW-FIT-0.5')
# plt.scatter(pha3, third(np.array([pha3]), 1.5, 1.555e-4, -0.042, 0.333, 197.18, 0), s=1, c='k', label='NEW-FIT-1.0')
# plt.scatter(pha3, third(np.array([pha3]), 1.0, 1.555e-4, -0.042, 0.333, 197.18, 0), s=1, c='k', label='NEW-FIT-1.5')
# plt.scatter(pha3, third(np.array([pha3]), 2.0, 1.555e-4, -0.042, 0.333, 197.18, 0), s=1, c='k', label='NEW-FIT-2.0')
# plt.scatter(pha3, third(np.array([pha3]), 0.053, 1.555e-4, -0.042, 0.333, 197.18, 0), s=1, c='k', label='NEW-FIT-0.053')

# plt.title('Phase Calibration \n $V_{ref} = 1000 mV$')
# plt.legend()
# plt.xticks([0, 30, 60, 90, 120, 150, 180])
# plt.yticks([-100, -50, 0, 50, 100])
# plt.grid()

# Amplitude Plot
plt.figure(figsize=(6, 6))
plt.scatter(amp0, a_vol0, s=5, label=r'1 kHz')
# plt.scatter(amp1, a_vol1, s=5, label=r'4 kHz')
# plt.scatter(amp2, a_vol2, s=5, label=r'7 kHz')
# plt.scatter(amp3, a_vol3, s=5, label=r'10 kHz')

plt.scatter(amp0, line(np.array([amp0]), 0.3009, 2.488), s=1, c='k', label='FIT-1kHz')
# plt.scatter(amp0, line(np.array([amp0]), 0.2063 * (201/209), 0), s=1, c='k', label='FIT-1kHz')
# plt.scatter(amp1, line(np.array([amp1]), 0.2063 * (201/209), 0), s=1, c='k', label='FIT-4kHz')
# plt.scatter(amp2, line(np.array([amp2]), 0.2063 * (201/209), 0), s=1, c='k', label='FIT-7kHz')
# plt.scatter(amp3, line(np.array([amp3]), 0.2063 * (201/209), 0), s=1, c='k', label='FIT-10kHz')

plt.title('Amplitude calibration')
plt.legend()
plt.xticks([0, 100, 200, 350, 400, 500])
plt.xlabel('Input amplitude ($V_{pp}$)')
plt.yticks([0, 50, 100, 150])
plt.ylabel('Output amplitude ($V_{dc}$)')
plt.grid()
plt.tight_layout()
plt.show()

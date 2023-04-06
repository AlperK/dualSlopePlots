from pathlib import Path as Path
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters, minimize, report_fit
from sklearn.metrics import r2_score


def sine_model(x, amp, a, freq, phi, offset):
    return amp * a * (np.sin(2 * np.pi * freq * (x + 90) + phi)) + np.log(amp) * offset


def line_model(x, slope, intercept):
    return x * slope + intercept


def sine_model_dataset(params, i, x):
    """calc sine from params for data set i
    using simple, hardwired naming convention"""
    amp = params['amp_%i' % (i + 1)].value
    a = params['a_%i' % (i + 1)].value
    freq = params['freq_%i' % (i + 1)].value
    phi = params['phi_%i' % (i + 1)].value
    offset = params['offset_%i' % (i + 1)].value
    return sine_model(x, amp, a, freq, phi, offset)


def objective(params, x, data):
    """ calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by Sine functions"""
    ndata, nx = data.shape
    resid = 0.0 * data[:]
    # make residual per data set
    for i in range(ndata):
        # resid[i, :] = data[i, :] - sine_model_dataset(params, i, x)
        resid[i, :] = data[i, :] - sine_model_dataset(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


date = '2022-12-06'
demodulator = 'Demodulator-1'
root = Path.joinpath(Path(date), Path(demodulator))
f = '1'
reference = ['500', '1000', '1500', '2000']
signal = ['500', '1000', '1500', '2000']

paths = [Path.joinpath(root, Path('PhaseCalibration'), Path(f'{f}kHz_{sig}mV_{ref}mV_phase.csv'))
         for sig in signal for ref in reference]

phases = np.array([np.loadtxt(str(path), delimiter=',').T for path in paths])
phases[:, 1] = phases[:, 1] * 1e3

# PHASE FITS
# print(phases[8])
# smodel = Model(sine_model)
# params = smodel.make_params(amp=0.4, a=301.17, freq=2.7773e-3, phi=0, offset=0)
# params['amp'].vary = False
# # params['a'].vary = False
# params['freq'].vary = False
# params['phi'].vary = False
# # params['offset'].vary = False
# sine1 = smodel.fit(phases[8][1][45:135], params, method='leastsq', x=phases[8][0][45:135])
# print(sine1.fit_report())

fit_params = Parameters()

# for iy, y in enumerate(phases[:, 1]):
#     amp = int(signal[int(iy/4) % 4])
#     print(amp)
#     fit_params.add('amp_%i' % (iy+1), value=amp)
#     fit_params['amp_%i' % (iy+1)].vary = False
#     fit_params.add('a_%i' % (iy+1), value=0.30040378, min=0.01,  max=2)
#     fit_params.add('freq_%i' % (iy+1), value=0.0027722)
#     fit_params['freq_%i' % (iy+1)].vary = False
#     fit_params.add('phi_%i' % (iy+1), value=0.0086891, min=-10, max=10)
#     # fit_params['phi_%i' % (iy+1)].vary = False
#     fit_params.add('offset_%i' % (iy+1), value=0.34910107, min=-1000, max=1000)
#
# for iy in (np.arange(2, 16)):
#     fit_params['a_%i' % iy].expr = 'a_1'
#     fit_params['freq_%i' % iy].expr = 'freq_1'
#     fit_params['phi_%i' % iy].expr = 'phi_1'
#     fit_params['offset_%i' % iy].expr = 'offset_1'

for iy, y in enumerate(phases[:, 1]):
    amp = int(signal[int(iy / 4) % 4])
    print(amp)
    fit_params.add('amp_%i' % (iy + 1), value=amp)
    fit_params['amp_%i' % (iy + 1)].vary = False
    fit_params.add('a_%i' % (iy + 1), value=0.3001, min=-2, max=2)
    fit_params['a_%i' % (iy + 1)].vary = False

    fit_params.add('freq_%i' % (iy + 1), value=0.00277764, min=-2, max=2)
    fit_params['freq_%i' % (iy + 1)].vary = False
    fit_params.add('phi_%i' % (iy + 1), value=0.0008, min=-2, max=2)
    fit_params.add('offset_%i' % (iy + 1), value=0.545, min=-10, max=10)
    fit_params['offset_%i' % (iy + 1)].vary = False

for iy in (np.arange(2, 16)):
    fit_params['a_%i' % iy].expr = 'a_1'
    fit_params['freq_%i' % iy].expr = 'freq_1'
    fit_params['phi_%i' % iy].expr = 'phi_1'
    fit_params['offset_%i' % iy].expr = 'offset_1'
result = minimize(objective, fit_params, method='leastsq', args=(phases[0][0], phases[:, 1]))
report_fit(result)
print()

r2_phase = []
for i in range(1):
    # y_fit = sine_model_dataset(fit_params, i, phases[0][0])
    y_fit = sine_model_dataset(fit_params, i, phases[0][0])
    # plt.plot(phases[0][0], phases[i, :][1], 'o', phases[0][0], y_fit, '-', label=f'{paths[i]}')
    plt.plot(phases[0][0], phases[i, :][1], 'o')
    plt.plot(phases[0][0], sine_model(phases[0][0],
                                      amp=int(signal[i % 4]),
                                      a=0.3001,
                                      freq=0.00277764,
                                      phi=0.0085,
                                      offset=0.545), '-',
             # label=f'{paths[i]}',
             )
    # plt.plot(phases[0][0], phases[i, :][1], 'o', color='r')
    r2_phase.append(r2_score(phases[i, :][1], sine_model(phases[0][0],
                                                         amp=int(signal[i % 4]),
                                                         a=0.3001,
                                                         freq=0.00277764,
                                                         phi=0.0085,
                                                         offset=0.545)))
plt.legend()
print(r2_phase)
print(f"r2 for phase: {np.array([r2_phase]).mean()}")
plt.show()


plt.figure()
for i in range(1):
    plt.plot(phases[0][0], phases[i, :][1], 'o--', label=f"{reference[i]} mV", alpha=0.2)
plt.xlabel("Phase (°)")
plt.ylabel("Demodulator DC output (mV)")
plt.title("Demodulator output vs Reference Amplitude")
plt.legend()
plt.grid()
plt.xticks([0, 30, 60, 90, 120, 150, 180])
plt.yticks(range(-150, 200, 50))

plt.figure()
for i in range(0, 4):
    if i != 2:
        plt.plot(phases[0][0], phases[i*4, :][1], 'o--', label=f"{signal[i]} mV", alpha=0.2)
    else:
        plt.plot(phases[0][0], phases[i*4, :][1], 'o--', label=f"{signal[i]} mV", alpha=1, markersize=10)


plt.xlabel("Phase (°)")
plt.ylabel("Demodulator DC output (mV)")
plt.title("Demodulator output vs Signal Amplitude")
plt.legend(prop={'size': 16})
plt.grid()
plt.xticks(range(0, 200, 30), fontsize=14)
plt.yticks(range(-600, 700, 200), fontsize=14)
plt.show()


demodulator = 'Demodulator-1'
root = Path.joinpath(Path(date), Path(demodulator))
amplitudes = np.loadtxt(Path.joinpath(root,
                                      Path('AmplitudeCalibration'),
                                      Path(f'{f}kHz_amplitude.csv'),
                                      ),
                        delimiter=','
                        ).T[:]
lModel = Model(line_model)
params = lModel.make_params(slope=3000, intercept=0)
result = lModel.fit(data=amplitudes[0][8:], x=amplitudes[1][8:] * 1000, slope=3000, intercept=0)
print(result.fit_report())

plt.figure()
plt.plot(amplitudes[0][8:], amplitudes[1][8:] * 1000, 'bo', label='Demodulator data', alpha=0.2)
plt.plot(line_model(amplitudes[1][8:] * 1000, slope=3.321, intercept=0.57), amplitudes[1][8:] * 1000,
         'k-', label='Line fit')
plt.ylabel("Demodulator output (mV)")
plt.xlabel("Input Amplitude (mV)")
plt.title("Demodulator output vs Signal Amplitude")
plt.legend()
plt.yticks(range(0, 700, 100))
plt.xticks(range(0, 2500, 500))
plt.grid()
plt.tight_layout()
r2_amplitude = r2_score(amplitudes[0][8:], line_model(amplitudes[1][8:] * 1000, slope=3.321, intercept=0.57))
print(f"r2 for amplitude fit: {r2_amplitude}")
plt.show()

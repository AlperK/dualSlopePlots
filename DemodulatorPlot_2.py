from pathlib import Path as Path
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters


def sine_model(x, amp, a, freq, phi, offset):
    return amp * a * (np.sin(2*np.pi*freq*(x+90) + phi)) + np.log(amp)*offset


def sine_model_dataset(params, i, x):
    """calc sine from params for data set i
    using simple, hardwired naming convention"""
    amp = params['amp_%i' % (i+1)].value
    a = params['a_%i' % (i+1)].value
    freq = params['freq_%i' % (i+1)].value
    phi = params['phi_%i' % (i+1)].value
    offset = params['offset_%i' % (i+1)].value
    return sine_model(x, amp, a, freq, phi, offset)


def objective(params, x, data):
    """ calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by Sine functions"""
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - sine_model_dataset(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


date = '2022-06-21'
demodulator = 'Demodulator-1'
root = Path.joinpath(Path(date), Path(demodulator))
f = '1'
reference = ['100', '400', '700', '1000']
signal = ['100', '400', '700', '1000']

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
for iy, y in enumerate(phases):
    fit_params.add( 'amp_%i' % (iy+1), value=0.5, min=0.0,  max=200)
    fit_params.add( 'amp_%i' % (iy+1), value=0.5, min=0.0,  max=200)
    fit_params.add( 'amp_%i' % (iy+1), value=0.5, min=0.0,  max=200)
    fit_params.add( 'cen_%i' % (iy+1), value=0.4, min=-2.0,  max=2.0)
    fit_params.add( 'sig_%i' % (iy+1), value=0.3, min=0.01, max=3.0)

plt.scatter(phases[8][0], sine_model(np.array([phases[8][0]]), amp=0.4, a=109.5, freq=0.00277, phi=0, offset=0.4),
            s=1, c='k', label='SINE-FIT-0.1')

plt.show()

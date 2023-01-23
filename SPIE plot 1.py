import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import json
from scipy.optimize import fsolve


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
        ac_3 = rolling_apply(fun=np.mean, a=ac.T[0][1][1], w=window)
        ac_4 = rolling_apply(fun=np.mean, a=ac.T[1][1][1], w=window)
        ph_1 = rolling_apply(fun=np.mean, a=ph.T[0][0][1], w=window)
        ph_2 = rolling_apply(fun=np.mean, a=ph.T[1][0][1], w=window)
        ph_3 = rolling_apply(fun=np.mean, a=ph.T[0][1][1], w=window)
        ph_4 = rolling_apply(fun=np.mean, a=ph.T[1][1][1], w=window)
    else:
        ac_1 = ac.T[0][0][1]
        ac_2 = ac.T[1][0][1]
        ph_1 = ph.T[0][0][1]
        ph_2 = ph.T[1][0][1]

    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Raw data')
    fig.suptitle(r'Raw AC and $\phi$ for both Pairs')

    ax = axes[0][0]
    ax.plot(ac_1,
            linewidth=2, color='brown')
    ax.set_title('Pair 1 AC at 35 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel('AC (mV)')

    ax = axes[0][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_1, w=window) / rolling_apply(fun=np.mean, a=ac_1, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 1 at 35 mm AC std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel('%')

    ax = axes[0][2]
    ax.plot(ph_1,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 1 $\phi$ at 35 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[0][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_1, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 1 at 35 mm $\phi$ std')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][0]
    ax.plot(ac_2,
            linewidth=2, color='brown')
    ax.set_title('Pair 1 AC at 25 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel('AC (mV)')

    ax = axes[1][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_2, w=window) / rolling_apply(fun=np.mean, a=ac_2, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 1 AC at 25 mm std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel('%')

    ax = axes[1][2]
    ax.plot(ph_2,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 1 $\phi$ at 25 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_2, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 1 at 25 mm $\phi$ std')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[2][0]
    ax.plot(ac_3,
            linewidth=2, color='brown')
    ax.set_title('Pair 2 AC at 25 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel('AC (mV)')

    ax = axes[2][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_3, w=window) / rolling_apply(fun=np.mean, a=ac_3, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 2 at 25 mm AC std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel('%')

    ax = axes[2][2]
    ax.plot(ph_3,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 2 $\phi$ at 25 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[2][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_3, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 2 at 25 mm $\phi$ std')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[3][0]
    ax.plot(ac_4,
            linewidth=2, color='brown')
    ax.set_title('Pair 2 AC at 35 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel('AC (mV)')

    ax = axes[3][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_4, w=window) / rolling_apply(fun=np.mean, a=ac_4, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 2 at 35 mm AC std / mean')
    ax.set_xlabel('Data count')
    ax.set_ylabel('%')

    ax = axes[3][2]
    ax.plot(ph_4,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 2 $\phi$ at 35 mm')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[3][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_4, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 2 at 35 mm $\phi$ std')
    ax.set_xlabel('Data count')
    ax.set_ylabel(r'$\phi$ (°)')

    fig.tight_layout()


def plot_optical_parameters(absorption, scattering, p_mua=None, p_mus=None, window=None):
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
    fig.canvas.manager.set_window_title('Optical Parameters')

    ax = axes[0][0]
    ax.scatter(np.arange(absorption.size), absorption,
               alpha=0.6, linewidth=2, color='brown', edgecolor='darkred')
    if p_mus is not None:
        ax.axhline(p_mua, color='black', alpha=1, linewidth=3, linestyle=':')
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
    if p_mus is not None:
        ax.axhline(p_mus, color='black', alpha=1, linewidth=3, linestyle=':')
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


def slope_equations_690(S, *data):
    RF = data[0]
    ua690 = data[1]
    us690 = data[2]

    Sac, Sp = S[0], S[1]
    # RF = 81.001e6
    # ua690 = 0.0124
    # us690 = 0.930

    w = 2 * np.pi * RF
    v = 3e11 / 1.4

    eq1 = (Sp / Sac - Sac / Sp) * (w / (2 * v))
    eq2 = (Sac ** 2 - Sp ** 2) / (3 * ua690)

    return [eq1 - ua690, eq2 - us690]
    # return [eq1, eq2]


def slope_equations_830(S, *data):
    RF = data[0]
    ua830 = data[1]
    us830 = data[2]

    Sac, Sp = S[0], S[1]
    # RF = 81.001e6
    # ua690 = 0.0124
    # us690 = 0.930

    w = 2 * np.pi * RF
    v = 3e11 / 1.4

    eq1 = (Sp / Sac - Sac / Sp) * (w / (2 * v))
    eq2 = (Sac ** 2 - Sp ** 2) / (3 * ua830)

    return [eq1 - ua830, eq2 - us830]
    # return [eq1, eq2]


date_830 = Path('2022-12-06')
date_690 = Path('2022-11-30')

measurement_830 = Path('DUAL-SLOPE-830')
measurement_690 = Path('DUAL-SLOPE-690-3')

measurementCount_830 = Path('2')
measurementCount_690 = Path('2')

location_830 = Path.joinpath(date_830, measurement_830, measurementCount_830)
location_690 = Path.joinpath(date_690, measurement_690, measurementCount_690)

amplitudeLocation_830 = Path.joinpath(location_830, Path('amplitude.csv'))
amplitudeLocation_690 = Path.joinpath(location_690, Path('amplitude.csv'))

phaseLocation_830 = Path.joinpath(location_830, Path('phase.csv'))
phaseLocation_690 = Path.joinpath(location_690, Path('phase.csv'))

p_ua830, p_us830 = 0.0077, 0.597
p_ua690, p_us690 = 0.0081, 0.761

amplitudes_830 = np.loadtxt(str(amplitudeLocation_830), delimiter=',')
amplitudes_830 = amplitudes_830.reshape((amplitudes_830.shape[0], 2, 2, 2))
amplitudes_690 = np.loadtxt(str(amplitudeLocation_690), delimiter=',')
amplitudes_690 = amplitudes_690.reshape((amplitudes_690.shape[0], 2, 2, 2))

phases_830 = np.loadtxt(str(phaseLocation_830), delimiter=',')
phases_830 = phases_830.reshape((phases_830.shape[0], 2, 2, 2))
phases_690 = np.loadtxt(str(phaseLocation_690), delimiter=',')
phases_690 = phases_690.reshape((phases_690.shape[0], 2, 2, 2))

separations = np.array(([
    [[25, 35], [35, 25]],
    [[25, 35], [35, 25]]
]))

with open(Path.joinpath(location_830, Path('measurement settings.json')), 'r') as f:
    settings_830 = json.load(f)
with open(Path.joinpath(location_690, Path('measurement settings.json')), 'r') as f:
    settings_690 = json.load(f)

frequency_830 = float(settings_830['RF']) * 1e6 + float(settings_830['IF']) * 1e3
frequency_690 = float(settings_690['RF']) * 1e6 + float(settings_690['IF']) * 1e3

window = 5
linearizedAmplitudes_830 = linearize_amplitudes(amplitudes_830, separations)
linearizedAmplitudes_690 = linearize_amplitudes(amplitudes_690, separations)

plot_raw_amplitude_phase(ac=amplitudes_830, ph=phases_830, window=window)
plot_raw_amplitude_phase(ac=amplitudes_690, ph=phases_690, window=window)

amplitude_slopes_830 = get_slopes(linearizedAmplitudes_830, separations)
amplitude_slopes_other_way_around_830 = \
    get_slopes(np.swapaxes(linearizedAmplitudes_830.reshape(linearizedAmplitudes_830.shape[0], 2, 2, 2), 3, 2),
               separations)
amplitude_slopes_690 = get_slopes(linearizedAmplitudes_690, separations)
amplitude_slopes_other_way_around_690 = \
    get_slopes(np.swapaxes(linearizedAmplitudes_690.reshape(linearizedAmplitudes_690.shape[0], 2, 2, 2), 3, 2),
               separations)

phase_slopes_830 = get_slopes(np.deg2rad(phases_830), separations)
phase_slopes_other_way_around_830 = \
    get_slopes(np.deg2rad(np.swapaxes(phases_830, 3, 2)), separations)
phase_slopes_690 = get_slopes(np.deg2rad(phases_690), separations)
phase_slopes_other_way_around_690 = \
    get_slopes(np.deg2rad(np.swapaxes(phases_690, 3, 2)), separations)


mu_a_830, mu_s_830 = get_optical_properties((amplitude_slopes_830.T[0][0][1] + amplitude_slopes_830.T[0][1][1]) / 2,
                                            (phase_slopes_830.T[0][0][1] + phase_slopes_830.T[0][1][1]) / 2,
                                            frequency_830)
mu_a_690, mu_s_690 = get_optical_properties((amplitude_slopes_690.T[0][0][1] + amplitude_slopes_690.T[0][1][1]) / 2,
                                            (phase_slopes_690.T[0][0][1] + phase_slopes_690.T[0][1][1]) / 2,
                                            frequency_690)

amplitude_slopes_830.T[0][0][1] = rolling_apply(fun=np.mean, a=amplitude_slopes_830.T[0][0][1], w=window)
amplitude_slopes_830.T[0][1][1] = rolling_apply(fun=np.mean, a=amplitude_slopes_830.T[0][1][1], w=window)
amplitude_slopes_690.T[0][0][1] = rolling_apply(fun=np.mean, a=amplitude_slopes_690.T[0][0][1], w=window)
amplitude_slopes_690.T[0][1][1] = rolling_apply(fun=np.mean, a=amplitude_slopes_690.T[0][1][1], w=window)

amplitude_slopes_other_way_around_830.T[0][0][1] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around_830.T[0][0][1], w=window)
amplitude_slopes_other_way_around_830.T[0][1][1] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around_830.T[0][1][1], w=window)
amplitude_slopes_other_way_around_690.T[0][0][1] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around_690.T[0][0][1], w=window)
amplitude_slopes_other_way_around_690.T[0][1][1] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around_690.T[0][1][1], w=window)

phase_slopes_830.T[0][0][1] = rolling_apply(fun=np.mean, a=phase_slopes_830.T[0][0][1], w=window)
phase_slopes_830.T[0][1][1] = rolling_apply(fun=np.mean, a=phase_slopes_830.T[0][1][1], w=window)
phase_slopes_690.T[0][0][1] = rolling_apply(fun=np.mean, a=phase_slopes_690.T[0][0][1], w=window)
phase_slopes_690.T[0][1][1] = rolling_apply(fun=np.mean, a=phase_slopes_690.T[0][1][1], w=window)

phase_slopes_other_way_around_830.T[0][0][1] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around_830.T[0][0][1], w=window)
phase_slopes_other_way_around_830.T[0][1][1] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around_830.T[0][1][1], w=window)
phase_slopes_other_way_around_690.T[0][0][1] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around_690.T[0][0][1], w=window)
phase_slopes_other_way_around_690.T[0][1][1] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around_690.T[0][1][1], w=window)

S_830 = fsolve(slope_equations_830, np.array([-0.1, 0.1]), args=(frequency_830, p_ua830, p_us830))
S_690 = fsolve(slope_equations_690, np.array([-0.1, 0.1]), args=(frequency_690, p_ua830, p_us690))

plot_optical_parameters(mu_a_830, mu_s_830, p_mua=p_ua830, p_mus=p_us830, window=window)
plot_optical_parameters(mu_a_690, mu_s_690, p_mua=p_ua690, p_mus=p_us690, window=window)

print(f'mu_a_830 error: {(np.mean(mu_a_830[~np.isnan(mu_a_830)]) - p_ua830) / np.mean(p_ua830) * 100}')
print(f'mu_s_830 error: {(np.mean(mu_s_830[~np.isnan(mu_s_830)]) - p_us830) / np.mean(p_us830) * 100}')
print(f'mu_a_690 error: {(np.mean(mu_a_690[~np.isnan(mu_a_690)]) - p_ua690) / np.mean(p_ua690) * 100}')
print(f'mu_s_690 error: {(np.mean(mu_s_690[~np.isnan(mu_s_690)]) - p_us690) / np.mean(p_us690) * 100}')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
fig.canvas.manager.set_window_title('Slopes')
fig.suptitle('Slope Plots')
ax = axes[0]
ax.set_title('Amplitude Slopes')
ax.set_ylabel('1/mm')
ax.set_xlabel('Data Count')
ax.plot(-amplitude_slopes_830.T[0][0][1], color='darkred', label='Pair 1')
ax.plot(-amplitude_slopes_830.T[0][1][1], color='darkslateblue', label='Pair 2')
ax.plot((-amplitude_slopes_830.T[0][0][1]+-amplitude_slopes_830.T[0][1][1])/2, color='black', label='Average')
ax.axhline(S_830[0], color='darkgreen', label='Expected')

ax = axes[1]
ax.set_title('Phase Slopes')
ax.set_ylabel('°/mm')
ax.set_xlabel('Data Count')
ax.plot(-phase_slopes_830.T[0][0][1]*180/np.pi, color='darkred')
ax.plot(-phase_slopes_830.T[0][1][1]*180/np.pi, color='darkslateblue')
ax.plot((-phase_slopes_830.T[0][0][1]+-phase_slopes_830.T[0][1][1])/2*180/np.pi, color='black')
ax.axhline(S_830[1]*180/np.pi, color='darkgreen')

fig.legend()
fig.tight_layout()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
fig.canvas.manager.set_window_title('Slopes')
fig.suptitle('Slope Plots the other way around')
ax = axes[0]
ax.set_title('Amplitude Slopes')
ax.set_ylabel('1/mm')
ax.set_xlabel('Data Count')
ax.plot(-amplitude_slopes_other_way_around_830.T[0][0][1], color='darkred', label='Pair 1')
ax.plot(-amplitude_slopes_other_way_around_830.T[0][1][1], color='darkslateblue', label='Pair 2')
ax.plot((-amplitude_slopes_other_way_around_830.T[0][0][1] +
         -amplitude_slopes_other_way_around_830.T[0][1][1])/2, color='black', label='Average')
ax.axhline(S_830[0], color='darkgreen', label='Expected')

ax = axes[1]
ax.set_title('Phase Slopes')
ax.set_ylabel('°/mm')
ax.set_xlabel('Data Count')
ax.plot(-phase_slopes_other_way_around_830.T[0][0][1]*180/np.pi, color='darkred')
ax.plot(-phase_slopes_other_way_around_830.T[0][1][1]*180/np.pi, color='darkslateblue')
ax.plot((-phase_slopes_other_way_around_830.T[0][0][1] +
         -phase_slopes_other_way_around_830.T[0][1][1])/2*180/np.pi, color='black')
ax.axhline(S_830[1]*180/np.pi, color='darkgreen')

fig.legend()
fig.tight_layout()


mu_a_830, mu_s_830 = get_optical_properties((amplitude_slopes_other_way_around_830.T[0][0][1] +
                                             amplitude_slopes_other_way_around_830.T[0][1][1]) / 2,
                                            (phase_slopes_other_way_around_830.T[0][0][1] +
                                             phase_slopes_other_way_around_830.T[0][1][1]) / 2,
                                            frequency_830)
mu_a_690, mu_s_690 = get_optical_properties((amplitude_slopes_other_way_around_690.T[0][0][1] +
                                             amplitude_slopes_other_way_around_690.T[0][1][1]) / 2,
                                            (phase_slopes_other_way_around_690.T[0][0][1] +
                                             phase_slopes_other_way_around_690.T[0][1][1]) / 2,
                                            frequency_690)

plot_optical_parameters(mu_a_830, mu_s_830, p_mua=p_ua830, p_mus=p_us830, window=window)
plot_optical_parameters(mu_a_690, mu_s_690, p_mua=p_ua690, p_mus=p_us690, window=window)


plt.close('all')

fig, axes = plt.subplots(ncols=2, figsize=(4, 3), sharex=True)

# ax = axes[0]
# divider = make_axes_locatable(ax)
# ax2 = divider.new_vertical(size='100%', pad=0.1)
# fig.add_axes(ax2)
axes[0].set_title('Amplitude slopes', fontsize=10)
axes[0].plot(-amplitude_slopes_other_way_around_690.T[0][0][1], color='blue', label='Pair 1')
axes[0].plot(-amplitude_slopes_other_way_around_690.T[0][1][1], color='red', label='Pair 2')
# axes[0].set_ylim([-0.30, -0.05])
# ax.spines['top'].set_visible(False)
axes[0].plot((-amplitude_slopes_690.T[0][0][1]+-amplitude_slopes_690.T[0][1][1])/2, color='black', label='Average')
# ax.plot((-amplitude_slopes_690.T[0][0][1]+-amplitude_slopes_690.T[0][1][1])/2, color='black', label='Average')
axes[0].axhline(S_690[0], color='green', label='Known')
# ax2.set_ylim([-0.20, 0])
# ax2.spines['bottom'].set_visible(False)

# # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
# ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
# kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
# ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
axes[1].set_title('Phase slopes', fontsize=10)
axes[1].plot(-phase_slopes_other_way_around_690.T[0][0][1], color='red', label='Pair 1')
axes[1].plot(-phase_slopes_other_way_around_690.T[0][1][1], color='blue', label='Pair 2')
axes[1].plot((-phase_slopes_690.T[0][0][1]+-phase_slopes_690.T[0][1][1])/2, color='black', label='Average')
axes[1].axhline(S_690[1], color='green')

axes[0].legend(
    # bbox_to_anchor=(0., -1.02, 2.5, -1.02),
    bbox_to_anchor=(0, -0.35, 2.5, -0.0),
    # bbox_to_anchor=bb,
    loc='lower left',
    ncol=2, mode="expand",
    borderaxespad=0.,
    prop={'size': 8}
)
fig.tight_layout()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))

axes[0][0].set_title(r'$\mu_a(mm^{-1})$', fontsize=10)
axes[0][0].plot(mu_a_690, color='red')
axes[0][0].axhline(p_ua690, color='green')
axes[0][0].set_ylim([0.00, 0.015])

axes[1][0].set_title(r'$\mu_s\'(mm^{-1})$', fontsize=10)
l1, = axes[1][0].plot(mu_s_690, color='red', label='690nm')
axes[1][0].axhline(p_us690, color='green')
axes[1][0].set_ylim([0.0, 1.5])

axes[0][1].set_title(r'$\mu_a(mm^{-1})$', fontsize=10)
axes[0][1].plot(mu_a_830, color='darkred')
axes[0][1].axhline(p_ua830, color='green')
axes[0][1].set_ylim([0.00, 0.015])

axes[1][1].set_title(r'$\mu_s\'(mm^{-1})$', fontsize=10)
l2, = axes[1][1].plot(mu_s_830, color='darkred', label='830nm')
axes[1][1].axhline(p_us830, color='green')
axes[1][1].set_ylim([0.0, 1.5])

# handles, labels = axes.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center',
           # bbox_to_anchor=(0, -0.3, 2.5, -0.5),
           # )
# plt.subplots_adjust(hspace=10)

# fig.legend()
plt.subplots_adjust(hspace=0)
axes[1][0].legend(
    handles=[l1, l2], labels=['690nm', '830nm'],
    # bbox_to_anchor=(0., -1.02, 2.5, -1.02),
    bbox_to_anchor=(0, -0.35, 2.5, -0.0),
    # bbox_to_anchor=bb,
    loc='lower left',
    ncol=2, mode="expand",
    borderaxespad=0.,
    prop={'size': 8}
)
fig.tight_layout()

fig, axes = plt.subplots(ncols=2, figsize=(4, 3), sharex=True)

axes[0].set_title(r'$\mu_a (mm^{-1})$', fontsize=10)
axes[0].plot(mu_a_690, color='darkred', label="Measured")
axes[0].axhline(p_ua690, color='green', label='Expected')
axes[0].set_xlabel("Data point")
axes[0].set_ylabel("$(mm^{-1})$")
axes[0].set_ylim(0, 0.01)

axes[1].set_title(r"$\mu'_s (mm^{-1})$", fontsize=10)
axes[1].plot(mu_s_690, color='darkred', label="Measured")
axes[1].axhline(p_us690, color='green')
axes[1].set_xlabel("Data point")
axes[1].set_ylabel("($mm^{-1})$")
axes[1].set_ylim(0, 1.0)

fig.legend(
    bbox_to_anchor=(0.2, -0.2, 0.68, 0.5),
    loc='center',
    ncol=3, mode="expand",
    borderaxespad=0.,
    prop={'size': 8}
)
fig.subplots_adjust(left=0.25, right=0.95, wspace=0.6, bottom=0.25)


x = amplitudes_690.T[1][0][1]
x = x[~np.isnan(x)]
print(f"Amplitude std/mean: {np.std(x)/np.mean(x)*100}")

x = amplitudes_690.T[1][1][1]
x = x[~np.isnan(x)]
print(f"Amplitude std/mean: {np.std(x)/np.mean(x)*100}")

x = phases_690.T[1][0][1]
x = x[~np.isnan(x)]
print(f"Phase std: {np.std(x)}°")

x = phases_690.T[1][1][1]
x = x[~np.isnan(x)]
print(f"Phase std: {np.std(x)}°")
plt.show()


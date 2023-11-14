import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import fsolve


def apply_kalman_1d(initial_estimate,
                    initial_error_estimate,
                    error_in_measurement,
                    meas):
    initial_kalman_gain = initial_error_estimate / (initial_error_estimate + error_in_measurement)
    previous_estimate = initial_estimate
    previous_error_estimate = initial_error_estimate
    kalman_gain = initial_kalman_gain

    estimates = np.array([])

    for mea in meas:
        if np.isnan(mea) == False:
            # print(mea)
            current_estimate = previous_estimate + kalman_gain * (mea - previous_estimate)
            estimates = np.append(estimates, [current_estimate])
            # print(f'currentEstimate = {current_estimate}')
            # print(estimates)
            current_error_estimate = (1 - kalman_gain) * previous_error_estimate
            # print(f'currentErrorEstimate = {current_error_estimate}')

            kalman_gain = current_error_estimate / (current_error_estimate + error_in_measurement)
            # print(f'kalmanGain = {kalman_gain}')
            previous_estimate = current_estimate

    return estimates


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

    return (apply_kalman_1d(0.0077, initial_error_estimate=10, error_in_measurement=10,
                            meas=absorption_coefficient),
            scattering_coefficient)
    # return absorption_coefficient, scattering_coefficient


def plot_raw_amplitude_phase(ac, ph, window=None):
    if window is not None:
        ac_1 = rolling_apply(fun=np.mean, a=ac.T[0][0][0], w=window)
        ac_2 = rolling_apply(fun=np.mean, a=ac.T[1][0][0], w=window)
        ac_3 = rolling_apply(fun=np.mean, a=ac.T[0][1][0], w=window)
        ac_4 = rolling_apply(fun=np.mean, a=ac.T[1][1][0], w=window)
        ph_1 = rolling_apply(fun=np.mean, a=ph.T[0][0][0], w=window)
        ph_2 = rolling_apply(fun=np.mean, a=ph.T[1][0][0], w=window)
        ph_3 = rolling_apply(fun=np.mean, a=ph.T[0][1][0], w=window)
        ph_4 = rolling_apply(fun=np.mean, a=ph.T[1][1][0], w=window)
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
    est = apply_kalman_1d(ac_1[4], 100, 1000, ac_1)
    # print(est)
    ax.plot(est, color='green')
    ax.set_title('Pair 1 AC at 35 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel('AC (mV)')

    ax = axes[0][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_1, w=window) / rolling_apply(fun=np.mean, a=ac_1, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 1 at 35 mm AC std / mean')
    ax.set_xlabel('Data point')
    ax.set_ylabel('%')

    ax = axes[0][2]
    ax.plot(ph_1,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 1 $\phi$ at 35 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[0][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_1, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 1 at 35 mm $\phi$ std')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][0]
    ax.plot(ac_2,
            linewidth=2, color='brown')
    ax.set_title('Pair 1 AC at 25 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel('AC (mV)')

    ax = axes[1][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_2, w=window) / rolling_apply(fun=np.mean, a=ac_2, w=window),
            linewidth=2, color='brown')
    ax.plot(apply_kalman_1d(0.1, 0.1, 10,
                            100 * rolling_apply(fun=np.std, a=ac_2, w=window) / rolling_apply(fun=np.mean, a=ac_2, w=window)),
            color='green')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 1 AC at 25 mm std / mean')
    ax.set_xlabel('Data point')
    ax.set_ylabel('%')

    ax = axes[1][2]
    ax.plot(ph_2,
            linewidth=2, color='darkslateblue')
    ax.plot(apply_kalman_1d(ph_2[4], 0.5, 10, ph_2),
            color='green')
    ax.set_title(r'Pair 1 $\phi$ at 25 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_2, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 1 at 25 mm $\phi$ std')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[2][0]
    ax.plot(ac_3,
            linewidth=2, color='brown')
    ax.set_title('Pair 2 AC at 25 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel('AC (mV)')

    ax = axes[2][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_3, w=window) / rolling_apply(fun=np.mean, a=ac_3, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 2 at 25 mm AC std / mean')
    ax.set_xlabel('Data point')
    ax.set_ylabel('%')

    ax = axes[2][2]
    ax.plot(ph_3,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 2 $\phi$ at 25 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[2][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_3, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 2 at 25 mm $\phi$ std')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[3][0]
    ax.plot(ac_4,
            linewidth=2, color='brown')
    ax.set_title('Pair 2 AC at 35 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel('AC (mV)')

    ax = axes[3][1]
    ax.plot(100 * rolling_apply(fun=np.std, a=ac_4, w=window) / rolling_apply(fun=np.mean, a=ac_4, w=window),
            linewidth=2, color='brown')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title('Pair 2 at 35 mm AC std / mean')
    ax.set_xlabel('Data point')
    ax.set_ylabel('%')

    ax = axes[3][2]
    ax.plot(ph_4,
            linewidth=2, color='darkslateblue')
    ax.set_title(r'Pair 2 $\phi$ at 35 mm')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[3][3]
    ax.plot(rolling_apply(fun=np.std, a=ph_4, w=window),
            linewidth=2, color='darkslateblue')
    ax.axhline(0.1, color='black', alpha=1, linewidth=3, linestyle=':')
    ax.set_title(r'Pair 2 at 35 mm $\phi$ std')
    ax.set_xlabel('Data point')
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


def Slope_Equations_690(S, *data):
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


def Slope_Equations_830(S, *data):
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


date = Path('2023-11-13')
measurement = Path('DUAL-SLOPE-830-3')
measurementCount = Path('2')
location = Path.joinpath(date, measurement, measurementCount)

amplitudeLocation = Path.joinpath(location, Path('amplitude.csv'))
phaseLocation = Path.joinpath(location, Path('phase.csv'))

p_ua690, p_us690 = 0.0081, 0.761
p_ua830, p_us830 = 0.0077, 0.597
# p_ua690, p_us690 = 0.0077, 0.597

amplitudes = np.loadtxt(str(amplitudeLocation), delimiter=',')
amplitudes = amplitudes.reshape((amplitudes.shape[0], 2, 2, 2))
# print(amplitudes)
phases = np.loadtxt(str(phaseLocation), delimiter=',')
phases = phases.reshape((phases.shape[0], 2, 2, 2))

separations = np.array(([
    [[25, 35], [35, 25]],
    [[25, 35], [35, 25]]
]))

with open(Path.joinpath(location, Path('measurement settings.json')), 'r') as f:
    settings = json.load(f)

frequency = float(settings['RF']) * 1e6 + float(settings['IF']) * 1e3
window = 5
linearizedAmplitudes = linearize_amplitudes(amplitudes, separations)

plot_raw_amplitude_phase(ac=amplitudes, ph=phases, window=window)

amplitude_slopes = get_slopes(linearizedAmplitudes, separations)
# print(amplitude_slopes)
amplitude_slopes_other_way_around = \
    get_slopes(np.swapaxes(linearizedAmplitudes.reshape(linearizedAmplitudes.shape[0], 2, 2, 2), 3, 2), separations)
# amplitude_slopes[0] -> 820-1
# amplitude_slopes[1] -> 690-1
# amplitude_slopes[2] -> 820-2
# amplitude_slopes[3] -> 690-2

phase_slopes = get_slopes(np.deg2rad(phases), separations)
phase_slopes_other_way_around = \
    get_slopes(np.deg2rad(np.swapaxes(phases, 3, 2)), separations)
# phase_slopes[0][0][0] -> 820-1
# phase_slopes[0][0][1] -> 690-1
# phase_slopes[0][1][0] -> 820-2
# phase_slopes[0][1][1] -> 690-2


# mu_a, mu_s = get_optical_properties((amplitude_slopes.T[0][0][1] + amplitude_slopes.T[0][1][1]) / 2,
#                                     (phase_slopes.T[0][0][1] + phase_slopes.T[0][1][1]) / 2,
#                                     frequency)

mu_a, mu_s = get_optical_properties((amplitude_slopes.T[0][0][0] + amplitude_slopes.T[0][1][0]) / 2,
                                    (phase_slopes.T[0][0][0] + phase_slopes.T[0][1][0]) / 2,
                                    frequency)

amplitude_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=amplitude_slopes.T[0][0][0], w=window)
amplitude_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=amplitude_slopes.T[0][1][0], w=window)

amplitude_slopes_other_way_around.T[0][0][0] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][0][0], w=window)
amplitude_slopes_other_way_around.T[0][1][0] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][1][0], w=window)

amplitude_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=amplitude_slopes.T[0][0][0], w=window)
amplitude_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=amplitude_slopes.T[0][1][0], w=window)

amplitude_slopes_other_way_around.T[0][0][0] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][0][0], w=window)
amplitude_slopes_other_way_around.T[0][1][0] = \
    rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][1][0], w=window)

phase_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][0][0], w=window)
phase_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][1][0], w=window)

phase_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][0][0], w=window)
phase_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][1][0], w=window)
phase_slopes_other_way_around.T[0][0][0] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][0][0], w=window)
phase_slopes_other_way_around.T[0][1][0] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][1][0], w=window)

phase_slopes_other_way_around.T[0][0][0] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][0][0], w=window)
phase_slopes_other_way_around.T[0][1][0] = \
    rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][1][0], w=window)

# S = fsolve(Slope_Equations_690, np.array([-0.1, 0.1]), args=(frequency, p_ua690, p_us690))
S = fsolve(Slope_Equations_830, np.array([-0.1, 0.1]), args=(frequency, p_ua830, p_us830))

# plot_optical_parameters(mu_a, mu_s, p_mua=p_ua690, p_mus=p_us690, window=window)
plot_optical_parameters(mu_a, mu_s, p_mua=p_ua830, p_mus=p_us830, window=window)
print(f'mu_a error: {(np.mean(mu_a[~np.isnan(mu_a)]) - p_ua830) / np.mean(p_ua830) * 100}')
print(f'mu_s error: {(np.mean(mu_s[~np.isnan(mu_s)]) - p_us830) / np.mean(p_us830) * 100}')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
fig.canvas.manager.set_window_title('Slopes')
fig.suptitle('Slope Plots')
ax = axes[0]
ax.set_title('Amplitude Slopes')
ax.set_ylabel('1/mm')
ax.set_xlabel('Data Point')
ax.plot(amplitude_slopes.T[0][0][0], color='darkred', label='Pair 1')
ax.plot(amplitude_slopes.T[0][1][0], color='darkslateblue', label='Pair 2')
ax.plot((amplitude_slopes.T[0][0][0]+amplitude_slopes.T[0][1][0])/2, color='black', label='Average')
ax.axhline(S[0], color='darkgreen', label='Expected')

ax = axes[1]
ax.set_title('Phase Slopes')
ax.set_ylabel('°/mm')
ax.set_xlabel('Data Point')
ax.plot(phase_slopes.T[0][0][0]*180/np.pi, color='darkred')
ax.plot(phase_slopes.T[0][1][0]*180/np.pi, color='darkslateblue')
ax.plot((phase_slopes.T[0][0][0]+phase_slopes.T[0][1][0])/2*180/np.pi, color='black')
ax.axhline(S[1]*180/np.pi, color='darkgreen')

fig.legend()
fig.tight_layout()
# print(linearizedAmplitudes[0])
# print('asd')
# print('asd')
# print('asd')
# print(phase_slopes[0])
# print('asd')
# print('asd')
# print('asd')
# print(phase_slopes.T[0][0][1])


# print(linearizedAmplitudes[0][1])
# print()
# print(linearizedAmplitudes[0][1].T)
# print()
# print(linearizedAmplitudes.T[0][1])
# print()
# print(linearizedAmplitudes.reshape(linearizedAmplitudes.shape[0], 2, 2, 2)[0])
# print()
# print(linearizedAmplitudes.reshape(linearizedAmplitudes.shape[0], 2, 2, 2)[0][1].T)
# print()
# print(np.swapaxes(linearizedAmplitudes.reshape(linearizedAmplitudes.shape[0], 2, 2, 2), 3, 2)[0])
# print()
# print(amplitude_slopes.T[0][0][1])
# print(amplitude_slopes.T[0][1][1])

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
# fig.canvas.manager.set_window_title('Slopes')
# fig.suptitle('Slope Plots the other way around')
# ax = axes[0]
# ax.set_title('Amplitude Slopes')
# ax.set_ylabel('1/mm')
# ax.set_xlabel('Data Point')
# ax.plot(-amplitude_slopes_other_way_around.T[0][0][1], color='darkred', label='Pair 1')
# ax.plot(-amplitude_slopes_other_way_around.T[0][1][1], color='darkslateblue', label='Pair 2')
# ax.plot((-amplitude_slopes_other_way_around.T[0][0][1] +
#          -amplitude_slopes_other_way_around.T[0][1][1])/2, color='black', label='Average')
# ax.axhline(S[0], color='darkgreen', label='Expected')

# ax = axes[1]
# ax.set_title('Phase Slopes')
# ax.set_ylabel('°/mm')
# ax.set_xlabel('Data Point')
# ax.plot(-phase_slopes_other_way_around.T[0][0][1]*180/np.pi, color='darkred')
# ax.plot(-phase_slopes_other_way_around.T[0][1][1]*180/np.pi, color='darkslateblue')
# ax.plot((-phase_slopes_other_way_around.T[0][0][1] +
#          -phase_slopes_other_way_around.T[0][1][1])/2*180/np.pi, color='black')
# ax.axhline(S[1]*180/np.pi, color='darkgreen')

fig.legend()
fig.tight_layout()


# mu_a, mu_s = get_optical_properties((amplitude_slopes_other_way_around.T[0][0][1] +
#                                      amplitude_slopes_other_way_around.T[0][1][1]) / 2,
#                                     (phase_slopes_other_way_around.T[0][0][1] +
#                                      phase_slopes_other_way_around.T[0][1][1]) / 2,
#                                     frequency)
# plot_optical_parameters(mu_a, mu_s, p_mua=p_ua690, p_mus=p_us690, window=window)


plt.show()

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


def get_slopes(amplitudes, phases, separations):
    # amplitude_slopes[0] -> 820-1
    # amplitude_slopes[2] -> 820-2
    # amplitude_slopes[1] -> 690-1
    # amplitude_slopes[3] -> 690-2

    # phase_slopes[0][0][0] -> 820-1
    # phase_slopes[0][0][1] -> 690-1
    # phase_slopes[0][1][0] -> 820-2
    # phase_slopes[0][1][1] -> 690-2

    # First linearize the amplitudes: ln(I*r*r))
    # then compute the slopes
    linearized_amplitudes = linearize_amplitudes(amplitudes, separations)
    amplitude_slopes = np.diff(linearized_amplitudes) / np.diff(separations)

    # First convert the phases into radians
    # then compute the slopes
    phases = np.deg2rad(phases)
    phase_slopes = np.diff(phases) / np.diff(separations)

    return amplitude_slopes, phase_slopes


def get_dual_slopes(amplitude_slopes, phase_slopes):
    dual_amplitude_slopes = amplitude_slopes.mean(axis=2)
    dual_amplitude_slopes_color_1 = dual_amplitude_slopes.T.squeeze()[0]
    dual_amplitude_slopes_color_2 = dual_amplitude_slopes.T.squeeze()[1]

    dual_phase_slopes = phase_slopes.mean(axis=2)
    dual_phase_slopes_color_1 = dual_phase_slopes.T.squeeze()[0]
    dual_phase_slopes_color_2 = dual_phase_slopes.T.squeeze()[1]

    # print(dual_phase_slopes.T.shape)
    # print(dual_amplitude_slopes.T.squeeze().shape)
    # print('--------------')
    # print(dual_phase_slopes.T.squeeze()[0])
    # print('--------------')

    print(f'dual_amplitude_slopes_color_1[5]: {dual_amplitude_slopes_color_1[5]}')
    print(f'dual_phase_slopes_color_1[5]: {dual_phase_slopes_color_1[5]}')
    return (dual_amplitude_slopes_color_1, dual_amplitude_slopes_color_2,
            dual_phase_slopes_color_1, dual_phase_slopes_color_2)


def get_optical_properties(s_ac, s_p, f):
    w = 2 * np.pi * f
    n = 1.4
    c = 2.998e11

    absorption_coefficient = (w / (2 * (c/n))) * (np.divide(s_p, s_ac) - np.divide(s_ac, s_p))

    scattering_coefficient = (np.square(s_ac) - np.square(s_p)) / (3 * absorption_coefficient)

    # return (apply_kalman_1d(0.0077, initial_error_estimate=10, error_in_measurement=10,
    #                         meas=absorption_coefficient),
    #         scattering_coefficient)
    return absorption_coefficient, scattering_coefficient


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
        ac_5 = rolling_apply(fun=np.mean, a=ac.T[0][0][1], w=window)
        ac_6 = rolling_apply(fun=np.mean, a=ac.T[1][0][1], w=window)
        ac_7 = rolling_apply(fun=np.mean, a=ac.T[0][1][1], w=window)
        ac_8 = rolling_apply(fun=np.mean, a=ac.T[1][1][1], w=window)
        ph_5 = rolling_apply(fun=np.mean, a=ph.T[0][0][1], w=window)
        ph_6 = rolling_apply(fun=np.mean, a=ph.T[1][0][1], w=window)
        ph_7 = rolling_apply(fun=np.mean, a=ph.T[0][1][1], w=window)
        ph_8 = rolling_apply(fun=np.mean, a=ph.T[1][1][1], w=window)
    else:
        ac_1 = ac.T[0][0][1]
        ac_2 = ac.T[1][0][1]
        ph_1 = ph.T[0][0][1]
        ph_2 = ph.T[1][0][1]

    t = np.linspace(0, 10 * 60, rolling_apply(fun=np.std, a=ac_1, w=window).size)
    fig, axes = plt.subplots(4, 4, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Raw data')
    fig.suptitle(r'Raw AC and $\phi$' 
                 "\n" 
                 '830nm: Left, 690nm: Right')

    ax = axes[0][0]
    ax.plot(t, ac_1,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 1 AC at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[0][1]
    ax.plot(t, ph_1,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 1 $\phi$ at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][0]
    ax.plot(t, ac_2,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 1 AC at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[1][1]
    ax.plot(t, ph_2,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 1 $\phi$ at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[2][0]
    ax.plot(t, ac_3,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 2 AC at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[2][1]
    ax.plot(t, ph_3,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 2 $\phi$ at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[3][0]
    ax.plot(t, ac_4,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 2 AC at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[3][1]
    ax.plot(t, ph_4,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 2 $\phi$ at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[0][2]
    ax.plot(t, ac_5,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 1 AC at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[0][3]
    ax.plot(t, ph_5,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 1 $\phi$ at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[1][2]
    ax.plot(t, ac_6,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 1 AC at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[1][3]
    ax.plot(t, ph_6,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 1 $\phi$ at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[2][2]
    ax.plot(t, ac_7,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 2 AC at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[2][3]
    ax.plot(t, ph_7,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 2 $\phi$ at 35 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')

    ax = axes[3][2]
    ax.plot(t, ac_8,
            linewidth=2, color='brown')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title('Pair 2 AC at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('AC (mV)')

    ax = axes[3][3]
    ax.plot(t, ph_8,
            linewidth=2, color='darkslateblue')
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    ax.set_title(r'Pair 2 $\phi$ at 25 mm')
    ax.set_xlabel('Time(s)')
    ax.set_ylabel(r'$\phi$ (°)')


    fig.tight_layout()


def plot_slopes(amplitude_slopes, phase_slopes,
                dual_amplitude_slopes_color1,
                dual_phase_slopes_color_1,
                dual_amplitude_slopes_color2,
                dual_phase_slopes_color_2):
    phantom_slopes_830 = fsolve(slope_equations_830, np.array([-0.1, 0.1]), args=(frequency, p_ua830, p_us830))
    phantom_slopes_690 = fsolve(slope_equations_690, np.array([-0.1, 0.1]), args=(frequency, p_ua690, p_us690))

    t = np.linspace(0, 10 * 60, rolling_apply(fun=np.std, a=amplitude_slopes.T[0][0][0], w=window).size)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    fig.canvas.manager.set_window_title('Slopes')
    fig.suptitle('Slope Plots')
    ax = axes[0][0]
    ax.set_title('Amplitude Slopes 830nm')
    ax.set_ylabel('1/mm')
    ax.set_xlabel('Time(s)')
    ax.plot(t, amplitude_slopes.T[0][0][0], color='darkred', label='Pair 1', linewidth=3, alpha=0.75)
    ax.plot(t, amplitude_slopes.T[0][1][0], color='darkslateblue', label='Pair 2', linewidth=3, alpha=0.75)
    ax.plot(t, dual_amplitude_slopes_color1, color='black', label='Average', linewidth=3, alpha=0.75)
    ax.axvspan(300, 480, alpha=0.15, color='green', label='arterial occlusion')
    # ax.axhline(phantom_slopes_830[0], color='darkgreen', label='Expected', linewidth=3, alpha=0.75)

    ax = axes[0][1]
    ax.set_title('Phase Slopes 830nm')
    ax.set_ylabel('°/mm')
    ax.set_xlabel('Time(s)')
    ax.plot(t, phase_slopes.T[0][0][0] * 180 / np.pi, color='darkred', linewidth=3, alpha=0.75)
    ax.plot(t, phase_slopes.T[0][1][0] * 180 / np.pi, color='darkslateblue', linewidth=3, alpha=0.75)
    ax.plot(t, dual_phase_slopes_color_1 * 180 / np.pi, color='black', linewidth=3, alpha=0.75)
    ax.axvspan(300, 480, alpha=0.15, color='green')
    # ax.axhline(phantom_slopes_830[1] * 180 / np.pi, color='darkgreen', linewidth=3, alpha=0.75)

    ax = axes[1][0]
    ax.set_title('Amplitude Slopes 690nm')
    ax.set_ylabel('1/mm')
    ax.set_xlabel('Time(s)')
    ax.plot(t, amplitude_slopes.T[0][0][1], color='darkred', linewidth=3, alpha=0.75)
    ax.plot(t, amplitude_slopes.T[0][1][1], color='darkslateblue', linewidth=3, alpha=0.75)
    ax.plot(t, dual_amplitude_slopes_color2, color='black', linewidth=3, alpha=0.75)
    ax.axvspan(300, 480, alpha=0.15, color='green')
    # ax.axhline(phantom_slopes_690[0], color='darkgreen', linewidth=3, alpha=0.75)

    ax = axes[1][1]
    ax.set_title('Phase Slopes 690nm')
    ax.set_ylabel('°/mm')
    ax.set_xlabel('Time(s)')
    ax.plot(t, phase_slopes.T[0][0][1] * 180 / np.pi, color='darkred', linewidth=3, alpha=0.75)
    ax.plot(t, phase_slopes.T[0][1][1] * 180 / np.pi, color='darkslateblue', linewidth=3, alpha=0.75)
    ax.plot(t, dual_phase_slopes_color_2 * 180 / np.pi, color='black', linewidth=3, alpha=0.75)
    ax.axvspan(300, 480, alpha=0.15, color='green')
    # ax.axhline(phantom_slopes_690[1] * 180 / np.pi, color='darkgreen', linewidth=3, alpha=0.75)
    fig.legend()
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
    fig.suptitle(r'Optical Parameters for 690nm')

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


# Set the path for the measurement folder
date = Path('2023-12-14')
measurement = Path('AO-5-3-2')
measurementCount = Path('2')
location = Path.joinpath(date, measurement, measurementCount)

# Set the paths for the amplitude and phase data
amplitudeLocation = Path.joinpath(location, Path('amplitude.csv'))
phaseLocation = Path.joinpath(location, Path('phase.csv'))

# Optical parameters of the phantom
p_ua690, p_us690 = 0.0081, 0.761
p_ua830, p_us830 = 0.0077, 0.597

# Load the amplitudes and reshape them into [n][2][2][2]
amplitudes = np.loadtxt(str(amplitudeLocation), delimiter=',')
amplitudes = amplitudes.reshape((amplitudes.shape[0], 2, 2, 2))
# print(amplitudes.size)
# print(amplitudes.shape)
# print(amplitudes.T[0][0][1])
# print(amplitudes.T[1][0][1])
# print(amplitudes.T[0][1][1])
# print(amplitudes.T[1][1][1])

# Load the phases and reshape them into [n][2][2][2]
phases = np.loadtxt(str(phaseLocation), delimiter=',')
phases = phases.reshape((phases.shape[0], 2, 2, 2))

# Set the source-detector separations
separations = np.array(([
    [[35, 25], [25, 35]],
    [[25, 35], [35, 25]]
]))

# Read the measurement settings for RF and IF
with open(Path.joinpath(location, Path('measurement settings.json')), 'r') as f:
    settings = json.load(f)
frequency = float(settings['RF']) * 1e6 + float(settings['IF']) * 1e3

amplitudeSlopes, phaseSlopes = get_slopes(amplitudes, phases, separations)
(dualAmplitudeSlopesColor1, dualAmplitudeSlopesColor2,
 dualPhaseSlopesColor1, dualPhaseSlopesColor2) = get_dual_slopes(amplitudeSlopes, phaseSlopes)
absorptionColor1, scatteringColor1 = get_optical_properties(dualAmplitudeSlopesColor1, dualPhaseSlopesColor1, frequency)
absorptionColor2, scatteringColor2 = get_optical_properties(dualAmplitudeSlopesColor2, dualPhaseSlopesColor2, frequency)
np.savetxt(Path.joinpath(location, '830nm-absorption.txt'), absorptionColor1)
np.savetxt(Path.joinpath(location, '690nm-absorption.txt'), absorptionColor2)
print(f'absorption 830nm: {absorptionColor1.mean()}')
print(f'scattering 830nm: {scatteringColor1.mean()}')
print(f'absorption 685nm: {absorptionColor2.mean()}')
print(f'scattering 685nm: {scatteringColor2.mean()}')

# amplitude_slopes_other_way_around = \
#     get_slopes(np.swapaxes(linearizedAmplitudes.reshape(linearizedAmplitudes.shape[0], 2, 2, 2), 3, 2), separations)
# phase_slopes_other_way_around = \
#     get_slopes(np.deg2rad(np.swapaxes(phases, 3, 2)), separations)


# mu_a, mu_s = get_optical_properties((amplitude_slopes.T[0][0][1] + amplitude_slopes.T[0][1][1]) / 2,
#                                     (phase_slopes.T[0][0][1] + phase_slopes.T[0][1][1]) / 2,
#                                     frequency)

# mu_a, mu_s = get_optical_properties((amplitudeSlopes.T[0][0][1] + amplitudeSlopes.T[0][1][1]) / 2,
#                                     (phaseSlopes.T[0][0][1] + phaseSlopes.T[0][1][1]) / 2,
#                                     frequency)

# amplitudeSlopes.T[0][0][0] = rolling_apply(fun=np.mean, a=amplitudeSlopes.T[0][0][0], w=window)
# amplitudeSlopes.T[0][1][0] = rolling_apply(fun=np.mean, a=amplitudeSlopes.T[0][1][0], w=window)

# amplitude_slopes_other_way_around.T[0][0][0] = \
#     rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][0][0], w=window)
# amplitude_slopes_other_way_around.T[0][1][0] = \
#     rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][1][0], w=window)

# amplitude_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=amplitude_slopes.T[0][0][0], w=window)
# amplitude_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=amplitude_slopes.T[0][1][0], w=window)

# amplitude_slopes_other_way_around.T[0][0][0] = \
#     rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][0][0], w=window)
# amplitude_slopes_other_way_around.T[0][1][0] = \
#     rolling_apply(fun=np.mean, a=amplitude_slopes_other_way_around.T[0][1][0], w=window)

# phase_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][0][0], w=window)
# phase_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][1][0], w=window)
#
# phase_slopes.T[0][0][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][0][0], w=window)
# phase_slopes.T[0][1][0] = rolling_apply(fun=np.mean, a=phase_slopes.T[0][1][0], w=window)
# phase_slopes_other_way_around.T[0][0][0] = \
#     rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][0][0], w=window)
# phase_slopes_other_way_around.T[0][1][0] = \
#     rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][1][0], w=window)
#
# phase_slopes_other_way_around.T[0][0][0] = \
#     rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][0][0], w=window)
# phase_slopes_other_way_around.T[0][1][0] = \
#     rolling_apply(fun=np.mean, a=phase_slopes_other_way_around.T[0][1][0], w=window)

# mu_a, mu_s = get_optical_properties((amplitude_slopes_other_way_around.T[0][0][1] +
#                                      amplitude_slopes_other_way_around.T[0][1][1]) / 2,
#                                     (phase_slopes_other_way_around.T[0][0][1] +
#                                      phase_slopes_other_way_around.T[0][1][1]) / 2,
#                                     frequency)
# plot_optical_parameters(mu_a, mu_s, p_mua=p_ua690, p_mus=p_us690, window=window)

# Plot the raw amplitude and phase

# Window size for the windowing during plotting
window = 1
plot_raw_amplitude_phase(ac=amplitudes, ph=phases, window=window)
plot_slopes(amplitude_slopes=amplitudeSlopes,
            phase_slopes=phaseSlopes,
            dual_amplitude_slopes_color1=dualAmplitudeSlopesColor1,
            dual_phase_slopes_color_1=dualPhaseSlopesColor1,
            dual_amplitude_slopes_color2=dualAmplitudeSlopesColor2,
            dual_phase_slopes_color_2=dualPhaseSlopesColor2,
            )
plot_optical_parameters(absorptionColor1, scatteringColor1, p_mua=p_ua830, p_mus=p_us830, window=window)
# print(f'mu_a error: {(np.mean(absorptionColor1[~np.isnan(absorptionColor1)]) - p_ua830) / np.mean(p_ua830) * 100}')
# print(f'mu_s error: {(np.mean(scatteringColor1[~np.isnan(scatteringColor1)]) - p_us690) / np.mean(p_us830) * 100}')


plt.show()

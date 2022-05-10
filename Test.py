import numpy as np


def sine_model(x, amp, a=201, freq=0.00278, phi=0, offset=1.4):
    # a = 201, freq = 0.00278, phi = 0, offset = 1.4
    return amp * a * (np.sin(2*np.pi*freq*(x+90) + phi)) + offset


def third(x, amp, a=1.6e-4, b=-0.043, c=0.26, d=208.5, e=0):
    # a = 1.6e-4, b = -0.043, c = 0.26, d = 208.5, e = 0
    return amp*(a*x**3 + b*x**2 + c*x + d) + e


def voltage2phase(phase_voltage, p):
    # print(phase_voltage)
    p[-1] = p[-1] - phase_voltage
    roots = np.roots(p)
    for root in roots:
        if (np.isreal(root)) and (root.real > 0) and (root.real < 180):
            return root


def voltage2phase4sine(phase_voltage, p):
    x = (phase_voltage - p[3]) / (p[0] * p[1])
    y = np.arcsin(x)
    phase = (y / (2 * np.pi * p[2])) - 90
    return phase


# Phase 19.202089579398727, Amplitude 0.05369436954677857, PhaseV 0.010672851562500404

print(sine_model(-30.81393, 0.05369))
print(third(-30.81393, 0.05369))
print(voltage2phase(10.672, 0.05369 * np.array([1.555e-4, -0.042, 0.333, 197.18])))
print(voltage2phase4sine(10.67285, np.array([0.05369, 201, 0.00278, 1.4])))

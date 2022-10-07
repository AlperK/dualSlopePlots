import numpy as np
from scipy.optimize import fsolve


def Slope_Equations_690(S):
    Sac, Sp = S[0], S[1]
    RF = 95e6
    ua690 = 0.0081
    us690 = 0.761

    w = 2 * np.pi * RF
    v = 3e11 / 1.4

    eq1 = (Sp / Sac - Sac / Sp) * (w / (2 * v))
    eq2 = (Sac ** 2 - Sp ** 2) / (3 * ua690)

    return [eq1 - ua690, eq2 - us690]
    # return [eq1, eq2]


def s_ac(u_a, u_s, f):
    # w = 2 * np.pi * f
    w = f
    v = 3e11 / 1.4

    temp = (3*u_s/2) * (-u_a + u_a*np.sqrt(1+(w**2/(v**2*u_a**2))))
    # temp *=

    return -temp**0.5


def s_ph(u_a, u_s, f):
    # w = 2 * np.pi * f
    w = f
    v = 3e11 / 1.4

    temp = (3 * u_s / 2) * (u_a + u_a*np.sqrt(1+(w**2/(v**2*u_a**2))))
    # temp = temp

    return temp**0.5


# Sac_690, Sp_690 = fsolve(Slope_Equations_690, [-0.1, 0.1], xtol=1e-6)
S = fsolve(Slope_Equations_690, np.array([-0.1, 0.1]))

print(S[0], S[1])
print(np.isclose(Slope_Equations_690(S), [0.0, 0.0]))

a = s_ac(u_a=0.0081, u_s=0.761, f=95e6)
s = s_ph(u_a=0.0081, u_s=0.761, f=95e6)
print(a, s)


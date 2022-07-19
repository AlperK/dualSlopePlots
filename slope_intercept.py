def get_slope(x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2)


def get_intercept(x1, y1, m):
    return y1 - m * x1


x1, y1 = 0.23, 0.7
x2, y2 = 0.397, 1.27

slope = get_slope(x1, y1, x2, y2)
intercept = get_intercept(x1, x2, slope)
print(f'For Demodulator 1:')
print(f'Slope: {slope}')
print(f'Intercept: {intercept}')


x1, y1 = 0.030, 0.08
x2, y2 = 0.0512, 0.18

slope = get_slope(x1, y1, x2, y2)
intercept = get_intercept(x1, x2, slope)
print(f'For Demodulator 2:')
print(f'Slope: {slope}')
print(f'Intercept: {intercept}')

x1, y1 = 80, 290
x2, y2 = 300, 950

slope = get_slope(x1, y1, x2, y2)
intercept = get_intercept(x1, y1, slope)
print(f'Demodulator-1 Slope {slope}')
print(f'Demodulator-1 Intercept {intercept}')
print(f'Demodulator-1 Amplitude {slope*x1 + intercept}')
print(f'Demodulator-1 Amplitude {slope*7.5 + intercept}')
print()

x1, y1 = 56.2, 185
x2, y2 = 159.5, 490

slope = get_slope(x1, y1, x2, y2)
intercept = get_intercept(x1, y1, slope)
print(f'Demodulator-2 Slope {slope}')
print(f'Demodulator-2 Intercept {intercept}')
print(f'Demodulator-2 Amplitude {slope*x1 + intercept}')
print(f'Demodulator-2 Amplitude {slope*199 + intercept}')
print()

import numpy as np
from matplotlib import pyplot as plt

def triangular(x, a1, a2, a3):
    if a1 < x < a2:
        return (x - a1) / (a2 - a1)
    if x == a2:
        return 1.0
    if a2 < x < a3:
        return (a3 - x) / (a3 - a2)
    return 0.0

def trapezoidal(x, a1, a2, a3, a4):
    if a1 < x < a2:
        return (x - a1) / (a2 - a1)
    if a2 <= x <= a3:
        return 1.0
    if a3 < x < a4:
        return (a4 - x) / (a4 - a3)
    return 0.0

if __name__ == '__main__':
    test_data = np.arange(-2.0, 6.0, 0.01)

    y_triangular = []
    for t in test_data:
        print('Dla x =', t, '  \t y =', triangular(t, 0, 2, 4))
        y_triangular.append(triangular(t, 0, 2, 4))

    print()

    y_trapezoidal = []
    for t in test_data:
        print('Dla x =', t, '  \t y =', trapezoidal(t, -1, 1, 3, 5))
        y_trapezoidal.append(trapezoidal(t, -1, 1, 3, 5))

    plt.plot(test_data, y_triangular)
    plt.plot(test_data, y_trapezoidal)
    plt.show()

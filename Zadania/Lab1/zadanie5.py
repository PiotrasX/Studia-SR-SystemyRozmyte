import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienna wejściowa 1
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'Odległość od najbliższego pojazdu [m]')

# Zmienna wejściowa 2
humidity_of_surface = ctrl.Antecedent(np.arange(0.0, 10.1, 0.1), 'Wilgotność nawierzchni drogowej')

# Zmienna wyjściowa
braking = ctrl.Consequent(np.arange(0.0, 6.1, 0.1), 'Siła hamowania')

# Funkcje przynależności do zmiennej wejściowej 1
distance['BARDZO DUŻA'] = fuzz.trapmf(distance.universe, [70, 90, 100, 100])
distance['DUŻA'] = fuzz.trapmf(distance.universe, [35, 50, 70, 90])
distance['ŚREDNIA'] = fuzz.trimf(distance.universe, [20, 35, 50])
distance['MAŁA'] = fuzz.trimf(distance.universe, [15, 25, 35])
distance['BARDZO MAŁA'] = fuzz.trapmf(distance.universe, [0, 0, 10, 25])
distance.view()

# Funkcje przynależności do zmiennej wejściowej 2
humidity_of_surface['BRAK'] = fuzz.trapmf(humidity_of_surface.universe, [0.0, 0.0, 0.1, 0.1])
humidity_of_surface['MAŁA'] = fuzz.trapmf(humidity_of_surface.universe, [0.0, 2.0, 4.0, 6.0])
humidity_of_surface['ŚREDNIA'] = fuzz.trapmf(humidity_of_surface.universe, [3.0, 5.0, 7.0, 8.0])
humidity_of_surface['DUŻA'] = fuzz.trapmf(humidity_of_surface.universe, [6.0, 8.0, 10.0, 10.0])
humidity_of_surface.view()

# Funkcje przynależności do zmiennej wyjściowej
braking['BRAK'] = fuzz.trapmf(braking.universe, [0.0, 0.0, 0.1, 0.1])
braking['MAŁA'] = fuzz.trapmf(braking.universe, [0.0, 1.0, 2.0, 3.0])
braking['ŚREDNIA'] = fuzz.trapmf(braking.universe, [2.0, 3.0, 4.0, 5.0])
braking['DUŻA'] = fuzz.trapmf(braking.universe, [4.0, 5.0, 6.0, 6.0])
braking.view()

# Reguły rozmyte
rules = [
    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['BRAK'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['BRAK'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['BRAK'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['BRAK'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['MAŁA'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['MAŁA'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['MAŁA'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['MAŁA'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['MAŁA'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['ŚREDNIA'], braking['MAŁA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['ŚREDNIA'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['ŚREDNIA'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['ŚREDNIA'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['ŚREDNIA'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['DUŻA'], braking['ŚREDNIA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['DUŻA'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['DUŻA'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['DUŻA'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['DUŻA'], braking['DUŻA']),
]

# Definicja sterownika rozmytego:
# Sterownik składa się z reguł rozmytych, ale same reguły
# składają się ze wcześniej zdefiniowanych wejść/wyjść
braking_ctrl = ctrl.ControlSystem(rules)

# Symulacja działania sterownika rozmytego
braking_simulation = ctrl.ControlSystemSimulation(braking_ctrl)

# Ustalenie różnych wejść ostrych (crisp)
distance_data = np.linspace(0, 100, 101)
humidity_of_surface_data = np.linspace(0, 10, 11)
X, Y = np.meshgrid(distance_data, humidity_of_surface_data)
Z = np.zeros_like(X, dtype=float)

# Zbieranie wyników dla każdej wartości ostrej
for hsd, sv in enumerate(humidity_of_surface_data):
    for dd, dv in enumerate(distance_data):
        braking_simulation.input['Odległość od najbliższego pojazdu [m]'] = float(np.asarray(dv).item()) # Ustawienie wejścia ostrego 1
        braking_simulation.input['Wilgotność nawierzchni drogowej'] = float(np.asarray(sv).item()) # Ustawienie wejścia ostrego 2
        braking_simulation.compute() # Przeprowadzenie symulacji
        Z[hsd, dd] = braking_simulation.output['Siła hamowania'] # Zapisanie wyniku symulacji
        braking_simulation.reset() # Resetowanie symulacji

# Wizualizacja siły hamowania na podstawie odległości od najbliższego pojazdu i wilgotności nawierzchni drogowej:

# Rysowanie osi 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

# Powierzchnia 3D
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.set_zlim(0, 6)
ax.set_xlabel('Odległość od najbliższego pojazdu [m]')
ax.set_ylabel('Wilgotność nawierzchni drogowej')
ax.set_zlabel('Siła hamowania')
ax.set_title('Siła hamowania na podstawie odległości od najbliższego pojazdu i wilgotności nawierzchni drogowej')

# Wizualizacja
plt.show()

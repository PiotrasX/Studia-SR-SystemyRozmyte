import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienna wejściowa
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'Odległość od najbliższego pojazdu [m]')

# Zmienna wyjściowa
braking = ctrl.Consequent(np.arange(0.0, 6.1, 0.1), 'Siła hamowania')

# Funkcje przynależności do zmiennej wejściowej
distance['BARDZO DUŻA'] = fuzz.trapmf(distance.universe, [70, 90, 100, 100])
distance['DUŻA'] = fuzz.trapmf(distance.universe, [35, 50, 70, 90])
distance['ŚREDNIA'] = fuzz.trimf(distance.universe, [20, 35, 50])
distance['MAŁA'] = fuzz.trimf(distance.universe, [15, 25, 35])
distance['BARDZO MAŁA'] = fuzz.trapmf(distance.universe, [0, 0, 10, 25])
distance.view()

# Funkcje przynależności do zmiennej wyjściowej
braking['BRAK'] = fuzz.trapmf(braking.universe, [0.0, 0.0, 0.1, 0.1])
braking['MAŁA'] = fuzz.trapmf(braking.universe, [0.0, 1.0, 2.0, 3.0])
braking['ŚREDNIA'] = fuzz.trapmf(braking.universe, [2.0, 3.0, 4.0, 5.0])
braking['DUŻA'] = fuzz.trapmf(braking.universe, [4.0, 5.0, 6.0, 6.0])
braking.view()

# Reguły rozmyte
rules = [
    ctrl.Rule(distance['BARDZO DUŻA'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'], braking['DUŻA'])
]

# Definicja sterownika rozmytego:
# Sterownik składa się z reguł rozmytych, ale same reguły
# składają się ze wcześniej zdefiniowanych wejść/wyjść
braking_ctrl = ctrl.ControlSystem(rules)

# Symulacja działania sterownika rozmytego
braking_simulation = ctrl.ControlSystemSimulation(braking_ctrl)

# Ustalenie różnych wejść ostrych (crisp)
distance_data = np.arange(0, 100.1, 0.1)
braking_out = []

# Zbieranie wyników dla każdej wartości ostrej
for dd in distance_data:
    braking_simulation.input['Odległość od najbliższego pojazdu [m]'] = dd # Ustawienie wejścia ostrego
    braking_simulation.compute() # Przeprowadzenie symulacji
    braking_out.append(braking_simulation.output['Siła hamowania']) # Zapisanie wyniku symulacji
    braking_simulation.reset() # Resetowanie symulacji

# Wizualizacja siły hamowania na podstawie odległości od najbliższego pojazdu
plt.figure()
plt.plot(distance_data, braking_out)
plt.xlim(0, 100)
plt.ylim(0, 6)
plt.xlabel('Odległość od najbliższego pojazdu [m]')
plt.ylabel('Siła hamowania')
plt.title('Siła hamowania na podstawie odległości od najbliższego pojazdu')
plt.grid(True)
plt.show()

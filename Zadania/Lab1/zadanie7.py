import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienne wejściowe
size = ctrl.Antecedent(np.arange(0, 11, 1), 'Rozmiar')
weight = ctrl.Antecedent(np.arange(0, 101, 1), 'Waga')

# Zmienna wyjściowa
quality = ctrl.Consequent(np.arange(0.0, 1.1, 0.1), 'Jakość owocu')

# Funkcje przynależności do zmiennej wejściowej 1
size['MAŁY'] = fuzz.trimf(size.universe, [0, 0, 10])
size['DUŻY'] = fuzz.trimf(size.universe, [0, 10, 10])
size.view()

# Funkcje przynależności do zmiennej wejściowej 2
weight['MAŁA'] = fuzz.trimf(weight.universe, [0, 0, 100])
weight['DUŻA'] = fuzz.trimf(weight.universe, [0, 100, 100])
weight.view()

# Funkcje przynależności do zmiennej wyjściowej
quality['SŁABA'] = fuzz.trimf(quality.universe, [0.0, 0.0, 0.5])
quality['ŚREDNIA'] = fuzz.trimf(quality.universe, [0.0, 0.5, 1.0])
quality['DOBRA'] = fuzz.trimf(quality.universe, [0.5, 1.0, 1.0])
quality.view()

# Reguły rozmyte
rule1 = ctrl.Rule(size['MAŁY'] & weight['MAŁA'], quality['SŁABA'])
rule2 = ctrl.Rule(size['MAŁY'] & weight['DUŻA'], quality['ŚREDNIA'])
rule3 = ctrl.Rule(size['DUŻY'] & weight['MAŁA'], quality['ŚREDNIA'])
rule4 = ctrl.Rule(size['DUŻY'] & weight['DUŻA'], quality['DOBRA'])

# Definicja sterownika rozmytego
quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

# Symulacja działania sterownika rozmytego
quality_simulation = ctrl.ControlSystemSimulation(quality_ctrl)

# Ustalenie wejść ostrych
size_value = 7
weight_value = 55
quality_simulation.input['Rozmiar'] = size_value
quality_simulation.input['Waga'] = weight_value

# Przeprowadzenie symulacji
quality_simulation.compute()

# Wizualizacja wyniku dla zmiennych wejściowych i zmiennej wyjściowej
size.view(sim=quality_simulation)
weight.view(sim=quality_simulation)
quality.view(sim=quality_simulation)

# Wypisanie wartości końcowej
print(f"Jakość owocu dla rozmiaru {size_value} i wagi {weight_value}:", quality_simulation.output['Jakość owocu'])

# Wyświetlenie wszystkich wykresów
plt.show()

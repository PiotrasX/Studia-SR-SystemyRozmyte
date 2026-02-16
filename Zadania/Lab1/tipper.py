import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienna wejściowa
quality_of_service = ctrl.Antecedent(np.arange(0, 11, 1), 'Jakość obsługi')

# Zmienna wyjściowa
tip = ctrl.Consequent(np.arange(0, 21, 1), 'Napiwek w %')

# Funkcje przynależności do zmiennej wejściowej
quality_of_service['ZŁA'] = fuzz.trapmf(quality_of_service.universe, [0, 0, 3, 5])
quality_of_service['ŚREDNIA'] = fuzz.trimf(quality_of_service.universe, [3, 5, 7])
quality_of_service['WYSOKA'] = fuzz.trapmf(quality_of_service.universe, [6, 7, 10, 10])
quality_of_service.view()

# Funkcje przynależności do zmiennej wyjściowej
tip['NISKI'] = fuzz.trapmf(tip.universe, [0, 0, 3, 5])
tip['ŚREDNI'] = fuzz.trimf(tip.universe, [4, 8, 12])
tip['WYSOKI'] = fuzz.trapmf(tip.universe, [10, 15, 20, 20])
tip.view()

# Reguły rozmyte
rule1 = ctrl.Rule(quality_of_service['ZŁA'], tip['NISKI'])
rule2 = ctrl.Rule(quality_of_service['ŚREDNIA'], tip['ŚREDNI'])
rule3 = ctrl.Rule(quality_of_service['WYSOKA'], tip['WYSOKI'])

# Definicja sterownika rozmytego:
# Sterownik składa się z reguł rozmytych, ale same reguły
# składają się ze wcześniej zdefiniowanych wejść/wyjść
tipper_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Symulacja działania sterownika rozmytego
tipper_simulation = ctrl.ControlSystemSimulation(tipper_ctrl)

# Ustalenie wejścia ostrego (crisp)
quality_value = 6.5
tipper_simulation.input['Jakość obsługi'] = quality_value

# Fuzzyfikacja wejścia ostrego — zamiana wartości ostrej na wejście rozmyte
# Podstawienie rozmytego wejścia do reguł
# Odczytanie z reguł rozmytego wyjścia
# Defuzzyfikacja zmiennej wyjściowej — zamiana wyjścia rozmytego na wartość ostrą
tipper_simulation.compute()

# Wizualizacja wyniku dla zmiennej wejściowej
quality_of_service.view(sim=tipper_simulation)

# Wizualizacja wyniku dla zmiennej wyjściowej
tip.view(sim=tipper_simulation)

# Wypisanie wartości końcowej (napiwek w %)
print(f"Napiwek dla jakości obsługi {quality_value}:", tipper_simulation.output['Napiwek w %'], "%")

# Wyświetlenie wszystkich wykresów
plt.show()

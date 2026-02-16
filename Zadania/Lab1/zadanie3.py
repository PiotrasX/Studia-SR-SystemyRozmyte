import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienna wejściowa 1
quality_of_service = ctrl.Antecedent(np.arange(0, 11, 1), 'Jakość obsługi')

# Zmienna wejściowa 2
quality_of_food = ctrl.Antecedent(np.arange(0, 11, 1), 'Jakość jedzenia')

# Zmienna wyjściowa
tip = ctrl.Consequent(np.arange(0, 21, 1), 'Napiwek w %')

# Funkcje przynależności do zmiennej wejściowej 1
quality_of_service['ZŁA'] = fuzz.trapmf(quality_of_service.universe, [0, 0, 3, 5])
quality_of_service['ŚREDNIA'] = fuzz.trimf(quality_of_service.universe, [3, 5, 7])
quality_of_service['WYSOKA'] = fuzz.trapmf(quality_of_service.universe, [6, 7, 10, 10])
quality_of_service.view()

# Funkcje przynależności do zmiennej wejściowej 2
quality_of_food['OHYDNA'] = fuzz.trapmf(quality_of_food.universe, [0, 0, 2, 4])
quality_of_food['ZNOŚNA'] = fuzz.trimf(quality_of_food.universe, [3, 6, 8])
quality_of_food['ZNAKOMITA'] = fuzz.trapmf(quality_of_food.universe, [5, 9, 10, 10])
quality_of_food.view()

# Funkcje przynależności do zmiennej wyjściowej
tip['NISKI'] = fuzz.trapmf(tip.universe, [0, 0, 3, 5])
tip['ŚREDNI'] = fuzz.trimf(tip.universe, [4, 8, 12])
tip['WYSOKI'] = fuzz.trapmf(tip.universe, [10, 15, 20, 20])
tip.view()

# Reguły rozmyte
rule1 = ctrl.Rule(quality_of_service['ZŁA'] & quality_of_food['OHYDNA'], tip['NISKI'])
rule2 = ctrl.Rule(quality_of_service['ZŁA'] & quality_of_food['ZNOŚNA'], tip['NISKI'])
rule3 = ctrl.Rule(quality_of_service['ZŁA'] & quality_of_food['ZNAKOMITA'], tip['ŚREDNI'])
rule4 = ctrl.Rule(quality_of_service['ŚREDNIA'] & quality_of_food['OHYDNA'], tip['NISKI'])
rule5 = ctrl.Rule(quality_of_service['ŚREDNIA'] & quality_of_food['ZNOŚNA'], tip['ŚREDNI'])
rule6 = ctrl.Rule(quality_of_service['ŚREDNIA'] & quality_of_food['ZNAKOMITA'], tip['WYSOKI'])
rule7 = ctrl.Rule(quality_of_service['WYSOKA'] & quality_of_food['OHYDNA'], tip['NISKI'])
rule8 = ctrl.Rule(quality_of_service['WYSOKA'] & quality_of_food['ZNOŚNA'], tip['WYSOKI'])
rule9 = ctrl.Rule(quality_of_service['WYSOKA'] & quality_of_food['ZNAKOMITA'], tip['WYSOKI'])

# Definicja sterownika rozmytego:
# Sterownik składa się z reguł rozmytych, ale same reguły
# składają się ze wcześniej zdefiniowanych wejść/wyjść
tipper_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Ustalenie różnych wejść ostrych (crisp)
quality_of_service_data = np.linspace(0, 10, 11)
quality_of_food_data = np.linspace(0, 10, 11)
X, Y = np.meshgrid(quality_of_service_data, quality_of_food_data)
Z = np.zeros_like(X, dtype=float)

# Zbieranie wyników dla każdej wartości ostrej
for qfd, fv in enumerate(quality_of_food_data):
    for qsd, sv in enumerate(quality_of_service_data):
        tipper_simulation = ctrl.ControlSystemSimulation(tipper_ctrl) # Symulacja działania sterownika rozmytego
        tipper_simulation.input['Jakość obsługi'] = float(np.asarray(sv).item()) # Ustawienie wejścia ostrego 1
        tipper_simulation.input['Jakość jedzenia'] = float(np.asarray(fv).item()) # Ustawienie wejścia ostrego 2
        tipper_simulation.compute() # Przeprowadzenie symulacji
        Z[qfd, qsd] = tipper_simulation.output['Napiwek w %'] # Zapisanie wyniku symulacji

# Wizualizacja zależności napiwku od jakości obsługi i jedzenia:

# Rysowanie osi 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

# Powierzchnia 3D
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 20)
ax.set_xlabel('Jakość obsługi')
ax.set_ylabel('Jakość jedzenia')
ax.set_zlabel('Napiwek w %')
ax.set_title('Wysokość napiwku na podstawie jakości obsługi i jedzenia')

# Wizualizacja
plt.show()

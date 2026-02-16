import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienna wejściowa 1
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'Odległość od najbliższego pojazdu [m]')

# Zmienna wejściowa 2
humidity_of_surface = ctrl.Antecedent(np.arange(0.0, 10.1, 0.1), 'Wilgotność nawierzchni drogowej')

# Zmienna wejściowa 3
icy_road_surface = ctrl.Antecedent(np.arange(0.0, 10.1, 0.1), 'Oblodzenie nawierzchni drogowej')

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

# Funkcje przynależności do zmiennej wejściowej 3
icy_road_surface['BRAK'] = fuzz.trapmf(icy_road_surface.universe, [0.0, 0.0, 0.1, 0.1])
icy_road_surface['MAŁE'] = fuzz.trapmf(icy_road_surface.universe, [0.0, 2.0, 3.0, 4.0])
icy_road_surface['ŚREDNIE'] = fuzz.trapmf(icy_road_surface.universe, [3.0, 4.0, 6.0, 7.0])
icy_road_surface['DUŻE'] = fuzz.trapmf(icy_road_surface.universe, [5.0, 7.0, 10.0, 10.0])
icy_road_surface.view()

# Funkcje przynależności do zmiennej wyjściowej
braking['BRAK'] = fuzz.trapmf(braking.universe, [0.0, 0.0, 0.1, 0.1])
braking['MAŁA'] = fuzz.trapmf(braking.universe, [0.0, 1.0, 2.0, 3.0])
braking['ŚREDNIA'] = fuzz.trapmf(braking.universe, [2.0, 3.0, 4.0, 5.0])
braking['DUŻA'] = fuzz.trapmf(braking.universe, [4.0, 5.0, 6.0, 6.0])
braking.view()

# Reguły rozmyte
rules = [
    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['BRAK'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['BRAK'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['BRAK'] & icy_road_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['BRAK'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['BRAK'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['BRAK'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['BRAK'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['MAŁA'] & icy_road_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['BRAK'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['BRAK'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['BRAK'], braking['MAŁA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['BRAK'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['BRAK'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['BRAK'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['DUŻA'] & icy_road_surface['BRAK'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['BRAK'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['BRAK'], braking['DUŻA']),



    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['MAŁE'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['MAŁE'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['BRAK'] & icy_road_surface['MAŁE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['MAŁE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['MAŁE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['MAŁE'], braking['BRAK']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['MAŁE'], braking['MAŁA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['MAŁA'] & icy_road_surface['MAŁE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['MAŁE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['MAŁE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['MAŁE'], braking['MAŁA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['MAŁE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['MAŁE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['MAŁE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['MAŁE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['MAŁE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['MAŁE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['DUŻA'] & icy_road_surface['MAŁE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['MAŁE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['MAŁE'], braking['DUŻA']),



    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['ŚREDNIE'], braking['MAŁA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['ŚREDNIE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['BRAK'] & icy_road_surface['ŚREDNIE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['ŚREDNIE'], braking['MAŁA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['ŚREDNIE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['MAŁA'] & icy_road_surface['ŚREDNIE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['ŚREDNIE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['ŚREDNIE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['DUŻA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['ŚREDNIE'], braking['DUŻA']),



    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['DUŻE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['BRAK'] & icy_road_surface['DUŻE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['BRAK'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['BRAK'] & icy_road_surface['DUŻE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['DUŻE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['MAŁA'] & icy_road_surface['DUŻE'], braking['ŚREDNIA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['MAŁA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['MAŁA'] & icy_road_surface['DUŻE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['ŚREDNIA'] & icy_road_surface['DUŻE'], braking['DUŻA']),

    ctrl.Rule(distance['BARDZO DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['DUŻA'] & humidity_of_surface['DUŻA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['ŚREDNIA'] & humidity_of_surface['DUŻA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
    ctrl.Rule(distance['BARDZO MAŁA'] & humidity_of_surface['DUŻA'] & icy_road_surface['DUŻE'], braking['DUŻA']),
]

# Definicja sterownika rozmytego:
# Sterownik składa się z reguł rozmytych, ale same reguły
# składają się ze wcześniej zdefiniowanych wejść/wyjść
braking_ctrl = ctrl.ControlSystem(rules)

# Symulacja działania sterownika rozmytego
braking_simulation = ctrl.ControlSystemSimulation(braking_ctrl)

# Ustalenie wejść ostrych (crisp)
distance_value = 30 # 0 <= x <= 100
humidity_of_surface_value = 5 # 0 <= x <= 10
icy_road_surface_value = 1 # 0 <= x <= 10
braking_simulation.input['Odległość od najbliższego pojazdu [m]'] = distance_value
braking_simulation.input['Wilgotność nawierzchni drogowej'] = humidity_of_surface_value
braking_simulation.input['Oblodzenie nawierzchni drogowej'] = icy_road_surface_value

# Przeprowadzenie symulacji
braking_simulation.compute()

# Wizualizacja wyniku dla zmiennej wejściowej 1
distance.view(sim=braking_simulation)

# Wizualizacja wyniku dla zmiennej wejściowej 2
humidity_of_surface.view(sim=braking_simulation)

# Wizualizacja wyniku dla zmiennej wejściowej 3
icy_road_surface.view(sim=braking_simulation)

# Wizualizacja wyniku dla zmiennej wyjściowej
braking.view(sim=braking_simulation)

# Wypisanie wartości końcowej
print(f"Odległość od najbliższego pojazdu [m]: {distance_value}")
print(f"Wilgotność nawierzchni drogowej: {humidity_of_surface_value}")
print(f"Oblodzenie nawierzchni drogowej: {icy_road_surface_value}")
print(f"\tSiła hamowania: {braking_simulation.output['Siła hamowania']}")

# Wyświetlenie wszystkich wykresów
plt.show()

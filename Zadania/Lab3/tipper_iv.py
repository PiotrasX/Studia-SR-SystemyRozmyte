import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyit2fls import Mamdani, min_t_norm, max_s_norm, IT2FS, tri_mf, ltri_mf, rtri_mf, TR_plot, crisp, IT2FS_plot

# ==============================
# Definiowanie uniwersów:
# Definiowanie zbiorów ostrych, dla których potem zdefiniuje się
# funkcje przynależności, które są zbiorami przedziałowo-rozmytymi.
# ==============================

# Uniwersum jakości jedzenia – skala od 0 do 10 punktów.
food_universe = np.linspace(0.0, 10.0, 1000)

# Uniwersum jakości obsługi – skala od 0 do 10 punktów.
service_universe = np.linspace(0.0, 10.0, 1000)

# Uniwersum napiwku – skala procentowa od 0 do 20 procent.
tip_universe = np.linspace(0.0, 20.0, 1000)

# ==============================
# Definiowanie zbiorów przedziałowo-rozmytych.
# ==============================

# Wyjaśnienie IT2FS:
# IT2FS(universe, upper_membership_function, upper_params, lower_membership_function, lower_params)
# * IT2FS — Konstruktor obiektu Interval Type-2 Fuzzy Set, czyli zbioru przedziałowo-rozmytego.
#   Ten konstruktor tworzy zbiór rozmyty typu II, który będzie opisany dwoma funkcjami przynależności (górną i dolną).
# * universe — Zakres wszystkich możliwych wartości zmiennej.
# * upper_membership_function, upper_params — Opis górnej funkcji przynależności.
#   upper_membership_function — Rodzaj funkcji przynależności.
#   upper_params — Parametry funkcji przynależności.
# * lower_membership_function, lower_params — Opis dolnej funkcji przynależności.
#   lower_membership_function — Rodzaj funkcji przynależności.
#   lower_params — Parametry funkcji przynależności.

# Przykłady dla membership_function oraz params:
# * rtri_mf [x1, x2, y]:
#       * x1 — Punkt, w którym funkcja zaczyna maleć (początek spadku),
#       * x2 — Punkt, w którym funkcja osiąga wartość 0 (koniec spadku),
#       * y — Maksymalna wartość, jaką przyjmie funkcja w punkcie x1.
#       * Funkcja:
#           - przyjmuje wartość y dla x ≤ x1,
#           - maleje liniowo od y do 0 przy x = x2,
#           - przyjmuje wartość 0 dla x ≥ x2.
# * ltri_mf [x1, x2, y]:
#       * x1 — Punkt, w którym funkcja zaczyna rosnąć (początek wzrostu),
#       * x2 — Punkt, w którym funkcja osiąga wartość y (koniec wzrostu),
#       * y — Maksymalna wartość, jaką przyjmie funkcja w punkcie x2.
#       * Funkcja:
#           - przyjmuje wartość 0 dla x ≤ x1,
#           - rośnie liniowo od 0 do y przy x = x2,
#           - przyjmuje wartość y dla x ≥ x2.
# * tri_mf [x1, x2, x3, y]:
#       * x1 — Punkt, w którym funkcja zaczyna rosnąć (początek wzrostu),
#       * x2 — Punkt, w którym funkcja osiąga wartość y (koniec wzrostu / środek trójkąta),
#              Punkt, w którym funkcja zaczyna maleć (początek spadku / środek trójkąta),
#       * x3 — Punkt, w którym funkcja osiąga wartość 0 (koniec spadku),
#       * y — Maksymalna wartość, jaką przyjmie funkcja w punkcie x2.
#       * Funkcja:
#           - przyjmuje wartość 0 dla x ≤ x1,
#           - rośnie liniowo od 0 do y dla x1 ≤ x ≤ x2 (w zakresie [x1, x2]),
#           - maleje liniowo od y do 0 dla x2 ≤ x ≤ x3 (w zakresie [x2, x3]),
#           - przyjmuje wartość 0 dla x ≥ x3.
# * trapezoid_mf [x1, x2, x3, x4, y]:
#       * x1 — Punkt, w którym funkcja zaczyna rosnąć (początek wzrostu),
#       * x2 — Punkt, w którym funkcja osiąga wartość y (koniec wzrostu / początek płaskiego wierzchołka),
#       * x3 — Punkt, w którym funkcja zaczyna maleć (początek spadku / koniec płaskiego wierzchołka),
#       * x4 — Punkt, w którym funkcja osiąga wartość 0 (koniec spadku),
#       * y — Maksymalna wartość, jaką przyjmie funkcja w zakresie [x2, x3].
#       * Funkcja:
#           - przyjmuje wartość 0 dla x ≤ x1,
#           - rośnie liniowo od 0 do y dla x1 ≤ x ≤ x2 (w zakresie [x1, x2]),
#           - utrzymuje wartość y (płaski szczyt) dla x2 ≤ x ≤ x3 (w zakresie [x2, x3]),
#           - maleje liniowo od y do 0 dla x3 ≤ x ≤ x4 (w zakresie [x3, x4]),
#           - przyjmuje wartość 0 dla x ≥ x4.

# Funkcje przynależności dla jakości jedzenia:

# "rancid" (zepsute) – prawa trójkątna funkcja przynależności ORAZ prawa trójkątna funkcja przynależności.
rancid_food = IT2FS(food_universe, rtri_mf, [3.01, 0, 1.0], rtri_mf, [2, 0, 0.99])
rancid_food.plot(title='Zepsute jedzenie')
rancid_food.check_set()

# "medium" (średnie) – trójkątna funkcja przynależności ORAZ trójkątna funkcja przynależności.
medium_food = IT2FS(food_universe, tri_mf, [2, 5.1, 8, 1.0], tri_mf, [4, 5.1, 6, 0.99])
medium_food.plot(title='Średnie jedzenie')
medium_food.check_set()

# "delicious" (pyszne) – lewa trójkątna funkcja przynależności ORAZ lewa trójkątna funkcja przynależności.
delicious_food = IT2FS(food_universe, ltri_mf, [7, 8, 1.0], ltri_mf, [8, 9, 0.99])
delicious_food.plot(title='Pyszne jedzenie')
delicious_food.check_set()

# Funkcje przynależności dla jakości obsługi:

# "low_service" (słaby) – prawa trójkątna funkcja przynależności ORAZ prawa trójkątna funkcja przynależności.
low_service = IT2FS(service_universe, rtri_mf, [3.01, 0, 1.0], rtri_mf, [2, 0, 0.99])
low_service.plot(title='Słaby serwis')
low_service.check_set()

# "medium_service" (średni) – trójkątna funkcja przynależności ORAZ trójkątna funkcja przynależności.
medium_service = IT2FS(service_universe, tri_mf,  [2, 5, 8, 1.0], tri_mf, [4, 5, 6, 1.0])
medium_service.plot(title='Średni serwis')
medium_service.check_set()

# "high_service" (dobry) – lewa trójkątna funkcja przynależności ORAZ lewa trójkątna funkcja przynależności.
high_service = IT2FS(service_universe, ltri_mf, [6, 8, 1.0], ltri_mf, [7, 9, 0.99])
high_service.plot(title='Dobry serwis')
high_service.check_set()

# Funkcje przynależności dla wysokości napiwku:

# "low_tip" (mały) – prawa trójkątna funkcja przynależności ORAZ prawa trójkątna funkcja przynależności.
low_tip = IT2FS(tip_universe, rtri_mf, [6, 0, 1.0], rtri_mf, [3, 0, 0.99])
low_tip.plot(title='Mały napiwek')
low_tip.check_set()

# "medium_tip" (średni) – trójkątna funkcja przynależności ORAZ trójkątna funkcja przynależności.
medium_tip = IT2FS(tip_universe, tri_mf, [4, 9, 14, 1.0], tri_mf, [8, 9, 10, 0.99])
medium_tip.plot(title='Średni napiwek')
medium_tip.check_set()

# "generous_tip" (hojny) – lewa trójkątna funkcja przynależności ORAZ lewa trójkątna funkcja przynależności.
generous_tip = IT2FS(tip_universe, ltri_mf, [12, 15, 1.0], ltri_mf, [15, 17, 0.99])
generous_tip.plot(title='Hojny napiwek')
generous_tip.check_set()

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
IT2FS_plot(rancid_food, medium_food, delicious_food, legends=["Zepsute", "Średnie", "Pyszne"], title='Jakość jedzenia')
IT2FS_plot(low_service, medium_service, high_service, legends=["Słaby", "Średni", "Dobry"], title='Jakość obsługi')
IT2FS_plot(low_tip, medium_tip, generous_tip, legends=["Mały", "Średni", "Hojny"], title='Wysokość napiwku [%]')
plt.show()

# ==============================
# Definiowanie sterownika rozmytego.
# ==============================

# Sterownik rozmyty typu "Mamdani" z metodą defuzyfikacji "Centroid".
controller = Mamdani(t_norm=min_t_norm, s_norm=max_s_norm, method="Centroid", algorithm="KM")

# Dwa wejścia i jedno wyjście dla sterownika rozmytego.
controller.add_input_variable('Jakość jedzenia')
controller.add_input_variable('Jakość obsługi')
controller.add_output_variable('Wysokość napiwku [%]')

# ==============================
# Definiowanie reguł rozmytych:
# Spójnik pomiędzy wejściami rozmytymi to AND.
# ==============================

controller.add_rule([('Jakość jedzenia', rancid_food), ('Jakość obsługi', low_service)], [('Wysokość napiwku [%]', low_tip)])
controller.add_rule([('Jakość jedzenia', rancid_food), ('Jakość obsługi', medium_service)], [('Wysokość napiwku [%]', medium_tip)])
controller.add_rule([('Jakość jedzenia', rancid_food), ('Jakość obsługi', high_service)], [('Wysokość napiwku [%]', medium_tip)])

controller.add_rule([('Jakość jedzenia', medium_food), ('Jakość obsługi', low_service)], [('Wysokość napiwku [%]', medium_tip)])
controller.add_rule([('Jakość jedzenia', medium_food), ('Jakość obsługi', medium_service)], [('Wysokość napiwku [%]', medium_tip)])
controller.add_rule([('Jakość jedzenia', medium_food), ('Jakość obsługi', high_service)], [('Wysokość napiwku [%]', generous_tip)])

controller.add_rule([('Jakość jedzenia', delicious_food), ('Jakość obsługi', low_service)], [('Wysokość napiwku [%]', medium_tip)])
controller.add_rule([('Jakość jedzenia', delicious_food), ('Jakość obsługi', medium_service)], [('Wysokość napiwku [%]', generous_tip)])
controller.add_rule([('Jakość jedzenia', delicious_food), ('Jakość obsługi', high_service)], [('Wysokość napiwku [%]', generous_tip)])

# ==============================
# Ocenianie systemu rozmytego dla konkretnych wartości.
# ==============================

# Przykładowe dane wejściowe (wejście ostre).
food_value = 0.0
service_value = 5.0
it2out, tr = controller.evaluate({"Jakość jedzenia": food_value, "Jakość obsługi": service_value})

# Rysowanie wykresu zbioru wynikowego.
it2out["Wysokość napiwku [%]"].plot()
TR_plot(tip_universe, tr["Wysokość napiwku [%]"])

# Wypisanie wartości po defuzyfikacji (wyjście ostre).
print(f'Napiwek w % dla jakości jedzenia {food_value} i jakości obsługi {service_value}: {crisp(tr['Wysokość napiwku [%]'])} %\n')

# ==============================
# Analiza wpływu poszczególnych zmiennych.
# ==============================

# Zakres zmiennych od 0 do 10 co 0,5 wartości.
x = np.arange(0, 10.5, 0.5)

# Wpływ jakości jedzenia (przy stałej jakości obsługi = 0).
tip = [crisp(controller.evaluate({"Jakość jedzenia": food, "Jakość obsługi": 0})[1]['Wysokość napiwku [%]']) for food in x]
print('Napiwek w zależności od jakości jedzenia (jakość obsługi = 0):')
print(tip, '\n')
plt.plot(x, tip)
plt.xlabel('Jakość jedzenia')
plt.ylabel('Wysokość napiwku [%]')
plt.title('Wpływ jakości jedzenia na napiwek (jakość obsługi = 0)')
plt.show()

# Wpływ jakości obsługi (przy stałej jakości jedzenia = 0).
tip = [crisp(controller.evaluate({"Jakość jedzenia": 0, "Jakość obsługi": service})[1]['Wysokość napiwku [%]']) for service in x]
print('Napiwek w zależności od jakości obsługi (jakość jedzenia = 0):')
print(tip, '\n')
plt.plot(x, tip)
plt.xlabel('Jakość obsługi')
plt.ylabel('Wysokość napiwku [%]')
plt.title('Wpływ jakości obsługi na napiwek (jakość jedzenia = 0)')
plt.show()

# ==============================
# Wizualizacja 3D dla napiwku na podstawie dwóch zmiennych.
# ==============================

# Zakres zmiennych od 0 do 10 co 0,25 wartości.
xs = np.arange(0, 10.25, 0.25)
ys = np.arange(0, 10.25, 0.25)

# Tablica z wartościami trójwymiarowymi (x, y, z), gdzie:
#   x – Jakość jedzenia,
#   y – Jakość obsługi,
#   z – Wysokość napiwku w %.
z = np.array([(x, y, crisp(controller.evaluate({'Jakość jedzenia': x, 'Jakość obsługi': y})[1]['Wysokość napiwku [%]'])) for x in xs for y in ys])
print('Tablica trójwymiarowa macierzy wyników:\n', z)

# Tworzenie wykresu 3D.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(z[:, 0], z[:, 1], z[:, 2])

ax.set_xlabel('Jakość jedzenia')
ax.set_ylabel('Jakość obsługi')
ax.set_zlabel('Wysokość napiwku [%]')
plt.title('Zależność napiwku od jakości jedzenia i jakości obsługi')
plt.show()

# ==============================
# Zapisanie wyników do pliku CSV.
# ==============================

# Zapis do pliku.
pd.DataFrame({
    'Jakość jedzenia': z[:, 0],
    'Jakość obsługi': z[:, 1],
    'Wysokość napiwku [%]': z[:, 2]
}).to_csv('tipper_iv.csv', index=False)

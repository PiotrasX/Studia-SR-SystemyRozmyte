import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyit2fls import T1TSK, T1FS, tri_mf, trapezoid_mf, T1FS_plot

# ==============================
# Definiowanie uniwersów:
# Definiowanie zbiorów ostrych, dla których potem zdefiniuje się
# funkcje przynależności wykorzystywane w modelu TSK.
# ==============================

# Uniwersum jakości jedzenia – skala od 0 do 10 punktów.
food_universe = np.linspace(0.0, 10.0, 1000)

# Uniwersum jakości obsługi – skala od 0 do 10 punktów.
service_universe = np.linspace(0.0, 10.0, 1000)

# ==============================
# Definiowanie zbiorów rozmytych typu I (T1FS).
# ==============================

# Wyjaśnienie T1FS:
# T1FS(universe, membership_function, params)
# * T1FS — Konstruktor obiektu Type-1 Fuzzy Set, czyli zbioru rozmytego typu I.
#   Ten konstruktor tworzy zbiór rozmyty typu I, który będzie opisany pojedynczą funkcją przynależności.
# * universe — Zakres wszystkich możliwych wartości zmiennej.
# * membership_function — Rodzaj funkcji przynależności.
# * params — Parametry funkcji przynależności.

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

# Dla zbiorów rozmytych T1FS zastosowano świadomie wykroczono poza uniwersum (np.: -1 i 11),
# ponieważ biblioteka nie pozwala na definiowanie dwóch takich samych wartości granicznych.

# Zbiory rozmyte dla jakości jedzenia:

# "rancid" (zepsute) – trapezoidalna funkcja przynależności.
rancid_food = T1FS(food_universe, trapezoid_mf, [-1, 0, 3, 5, 1.0])
rancid_food.plot('Zepsute jedzenie')

# "medium" (średnie) – trójkątna funkcja przynależności.
medium_food = T1FS(food_universe, tri_mf, [3, 5, 7, 1.0])
medium_food.plot('Średnie jedzenie')

# "delicious" (pyszne) – trapezoidalna funkcja przynależności.
delicious_food = T1FS(food_universe, trapezoid_mf, [6, 7, 10, 11, 1.0])
delicious_food.plot('Pyszne jedzenie')

# Zbiory rozmyte dla jakości obsługi:

# "low_service" (słaby) – trapezoidalna funkcja przynależności.
low_service = T1FS(service_universe, trapezoid_mf, [-1, 0, 3, 5, 1.0])
low_service.plot('Słaby serwis')

# "medium_service" (średni) – trójkątna funkcja przynależności.
medium_service = T1FS(service_universe, tri_mf, [3, 5, 7, 1.0])
medium_service.plot('Średni serwis')

# "high_service" (dobry) – trapezoidalna funkcja przynależności.
high_service = T1FS(service_universe, trapezoid_mf, [6, 7, 10, 11, 1.0])
high_service.plot('Dobry serwis')

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
T1FS_plot(rancid_food, medium_food, delicious_food, legends=["Zepsute", "Średnie", "Pyszne"], title='Jakość jedzenia')
T1FS_plot(low_service, medium_service, high_service, legends=["Słaba", "Średnia", "Dobra"], title='Jakość obsługi')

# ==============================
# Definiowanie funkcji wyjściowych (konsekwentów reguł TSK).
# ==============================

# Funkcje liniowe dwuargumentowe reprezentują wyjścia systemu TSK.
# W tym przypadku:
#   * napiwek = a0 + a1 * x1 + a2 * x2,
#   * gdzie:
#       * x1 – jakość jedzenia,
#       * x2 – jakość obsługi.

# Funkcja wyjściowa dla małego napiwku.
def low_tip(x1, x2):
    return 0 + 0.5 * x1 + 0.5 * x2

# Funkcja wyjściowa dla średniego napiwku.
def medium_tip(x1, x2):
    return 0 + 0.7 * x1 + 0.7 * x2

# Funkcja wyjściowa dla dużego napiwku.
def generous_tip(x1, x2):
    return 0 + x1 + x2

# ==============================
# Definiowanie sterownika rozmytego TSK.
# ==============================

# Sterownik rozmyty typu "TSK".
controller = T1TSK()

# Dwa wejścia i jedno wyjście dla sterownika rozmytego.
controller.add_input_variable('Jakość jedzenia')
controller.add_input_variable('Jakość obsługi')
controller.add_output_variable('Wysokość napiwku [%]')

# ==============================
# Definiowanie reguł rozmytych TSK:
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
# Ocenianie systemu rozmytego TSK dla konkretnych wartości.
# ==============================

# Przykładowe dane wejściowe (wejście ostre).
food_value = 0.0
service_value = 5.0
it2out = controller.evaluate({"Jakość jedzenia": food_value, "Jakość obsługi": service_value}, (food_value, service_value))

# Wypisanie wartości po defuzyfikacji (wyjście ostre).
print(f'Napiwek w % dla jakości jedzenia {food_value} i jakości obsługi {service_value}: {it2out['Wysokość napiwku [%]']} % \n')

# Wyświetlenie wszystkich wykresów.
plt.show()

# ==============================
# Analiza wpływu poszczególnych zmiennych.
# ==============================

# Zakres zmiennych od 0 do 10 co 0,5 wartości.
x = np.arange(0, 10.5, 0.5)

# Wpływ jakości jedzenia (przy stałej jakości obsługi = 0).
tip = [controller.evaluate({"Jakość jedzenia": food, "Jakość obsługi": 0}, (food, 0))['Wysokość napiwku [%]'] for food in x]
print('Napiwek w zależności od jakości jedzenia (jakość obsługi = 0):')
print(tip, '\n')
plt.plot(x, tip)
plt.xlabel('Jakość jedzenia')
plt.ylabel('Wysokość napiwku [%]')
plt.title('Wpływ jakości jedzenia na napiwek (jakość obsługi = 0)')
plt.show()

# Wpływ jakości obsługi (przy stałej jakości jedzenia = 0).
tip = [controller.evaluate({"Jakość jedzenia": 0, "Jakość obsługi": service}, (0, service))['Wysokość napiwku [%]'] for service in x]
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
z = np.array([(x, y, controller.evaluate({'Jakość jedzenia': x, 'Jakość obsługi': y}, (x, y))['Wysokość napiwku [%]']) for x in xs for y in ys])
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
}).to_csv('tipper_tsk.csv', index=False)

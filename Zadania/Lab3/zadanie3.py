import matplotlib.pyplot as plt
import numpy as np
from pyit2fls import T1TSK, T1FS, tri_mf, trapezoid_mf, T1FS_plot

# ==============================
# Definiowanie uniwersów.
# ==============================

# Uniwersum metrażu mieszkania – skala od 0 do 250 metrów kwadratowych.
square_footage_universe = np.linspace(0.0, 250.0, 2500)

# Uniwersum odległości mieszkania od ścisłego centrum – skala od 0 do 30 kilometrów.
distance_from_city_center_universe = np.linspace(0.0, 30.0, 300)

# Uniwersum wieku mieszkania – skala od 0 do 50 lat.
age_universe = np.linspace(0.0, 50.0, 500)

# ==============================
# Definiowanie zbiorów rozmytych typu I (T1FS).
# ==============================

# Zbiory rozmyte dla metrażu mieszkania:

small_apartment = T1FS(square_footage_universe, trapezoid_mf, [-1, 0, 25, 35, 1.0])
small_apartment.plot('Małe mieszkanie')

medium_apartment = T1FS(square_footage_universe, trapezoid_mf, [30, 50, 65, 75, 1.0])
medium_apartment.plot('Średnie mieszkanie')

large_apartment = T1FS(square_footage_universe, trapezoid_mf, [70, 100, 150, 185, 1.0])
large_apartment.plot('Duże mieszkanie')

giant_apartment = T1FS(square_footage_universe, trapezoid_mf, [170, 200, 250, 251, 1.0])
giant_apartment.plot('Gigantyczne mieszkanie')

# Zbiory rozmyte dla odległości mieszkania od ścisłego centrum:

short_distance = T1FS(distance_from_city_center_universe, tri_mf, [-1, 0, 5, 1.0])
short_distance.plot('Mała odległość')

medium_distance = T1FS(distance_from_city_center_universe, trapezoid_mf, [3, 7, 12, 15, 1.0])
medium_distance.plot('Średnia odległość')

long_distance = T1FS(distance_from_city_center_universe, trapezoid_mf, [13, 17, 30, 31, 1.0])
long_distance.plot('Duża odległość')

# Zbiory rozmyte dla wieku mieszkania:

young_age_apartment = T1FS(age_universe, trapezoid_mf, [-1, 0, 3, 7, 1.0])
young_age_apartment.plot('Młode mieszkanie')

middle_age_apartment = T1FS(age_universe, trapezoid_mf, [5, 9, 15, 22, 1.0])
middle_age_apartment.plot('Średniego wieku mieszkanie')

old_age_apartment = T1FS(age_universe, trapezoid_mf, [20, 30, 50, 51, 1.0])
old_age_apartment.plot('Stare mieszkanie')

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
T1FS_plot(small_apartment, medium_apartment, large_apartment, giant_apartment, legends=["Małe", "Średnie", "Duże", "Gigantyczne"], title='Metraż mieszkania')
T1FS_plot(short_distance, medium_distance, long_distance, legends=["Mała", "Średnia", "Duża"], title='Odległość mieszkania od ścisłego centrum')
T1FS_plot(young_age_apartment, middle_age_apartment, old_age_apartment, legends=["Młode", "Średniego wieku", "Stare"], title='Wiek mieszkania')

# ==============================
# Definiowanie funkcji wyjściowych (konsekwentów reguł TSK).
# ==============================

# Funkcja wyjściowa dla małej wartości mieszkania.
def small_value(size, distance, age):
    if size == 0: return 0
    return size * 7000 + (30 - distance) * 500 + (50 - age) * 900

# Funkcja wyjściowa dla średniej wartości mieszkania.
def medium_value(size, distance, age):
    if size == 0: return 0
    return size * 9000 + (30 - distance) * 1300 + (50 - age) * 1200

# Funkcja wyjściowa dla dużej wartości mieszkania.
def large_value(size, distance, age):
    if size == 0: return 0
    return size * 11000 + (30 - distance) * 2000 + (50 - age) * 1500

# ==============================
# Definiowanie sterownika rozmytego TSK.
# ==============================

# Sterownik rozmyty typu "TSK".
controller = T1TSK()

# Trzy wejścia rozmyte i jedno wyjście dla sterownika rozmytego.
controller.add_input_variable('Metraż mieszkania')
controller.add_input_variable('Odległość mieszkania od ścisłego centrum')
controller.add_input_variable('Wiek mieszkania')
controller.add_output_variable('Wartość mieszkania [zł]')

# ==============================
# Definiowanie reguł rozmytych TSK.
# ==============================

controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])

controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])

controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_value)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])

controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', large_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_value)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_value)])

# ==============================
# Ocenianie systemu rozmytego TSK dla konkretnych wartości.
# ==============================

# Przykładowe dane wejściowe (wejście ostre).
square_footage_values = np.arange(10, 260, 10)
distance_from_city_center_values = np.arange(0, 35, 5)
age_values = np.arange(0, 55, 5)
for sf_value in square_footage_values:
    for dfcc_value in distance_from_city_center_values:
        for a_value in age_values:
            it2out = controller.evaluate(({"Metraż mieszkania": sf_value, "Odległość mieszkania od ścisłego centrum": dfcc_value, "Wiek mieszkania": a_value}), (sf_value, dfcc_value, a_value))
            print(f'Cena mieszkania dla {sf_value} metrów kwadratowych, {dfcc_value} kilometrów odległości od ścisłego centrum, {a_value} lat wieku mieszkania: {it2out['Wartość mieszkania [zł]']} zł')
        if dfcc_value != 30: print()
    if sf_value != 250: print()

# Wyświetlenie wszystkich wykresów.
plt.show()

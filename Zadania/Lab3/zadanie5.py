import matplotlib.pyplot as plt
import numpy as np
from pyit2fls import Mamdani, min_t_norm, max_s_norm, IT2FS, tri_mf, IT2FS_plot, trapezoid_mf, crisp

# ==============================
# Definiowanie uniwersów.
# ==============================

# Uniwersum metrażu mieszkania – skala od 0 do 250 metrów kwadratowych.
square_footage_universe = np.linspace(0.0, 250.0, 2500)

# Uniwersum odległości mieszkania od ścisłego centrum – skala od 0 do 30 kilometrów.
distance_from_city_center_universe = np.linspace(0.0, 30.0, 300)

# Uniwersum wieku mieszkania – skala od 0 do 50 lat.
age_universe = np.linspace(0.0, 50.0, 500)

# Uniwersum ceny mieszkania – skala od 0 do 3500000 zł.
price_universe = np.linspace(0.0, 3500000.0, 35000)

# ==============================
# Definiowanie zbiorów przedziałowo-rozmytych.
# ==============================

# Funkcje przynależności dla metrażu mieszkania:

small_apartment = IT2FS(square_footage_universe, trapezoid_mf, [-1, 0, 25, 35, 1.0], trapezoid_mf, [-1, 0, 15, 25, 0.999])
small_apartment.plot('Małe mieszkanie')
small_apartment.check_set()

medium_apartment = IT2FS(square_footage_universe, trapezoid_mf, [30, 50, 65, 75, 1.0], tri_mf, [40, 57.5, 65, 0.75])
medium_apartment.plot('Średnie mieszkanie')
medium_apartment.check_set()

large_apartment = IT2FS(square_footage_universe, trapezoid_mf, [70, 100, 150, 185, 1.0], trapezoid_mf, [90, 110, 130, 165, 0.999])
large_apartment.plot('Duże mieszkanie')
large_apartment.check_set()

giant_apartment = IT2FS(square_footage_universe, trapezoid_mf, [170, 200, 250, 251, 1.0], trapezoid_mf, [190, 210, 250, 251, 0.999])
giant_apartment.plot('Gigantyczne mieszkanie')
giant_apartment.check_set()

# Funkcje przynależności dla odległości mieszkania od ścisłego centrum:

short_distance = IT2FS(distance_from_city_center_universe, tri_mf, [-1, 0, 5, 1.0], tri_mf, [-1, 0, 3, 0.95])
short_distance.plot('Mała odległość')
short_distance.check_set()

medium_distance = IT2FS(distance_from_city_center_universe, trapezoid_mf, [3, 7, 12, 15, 1.0], trapezoid_mf, [5, 9, 10, 13, 0.75])
medium_distance.plot('Średnia odległość')
medium_distance.check_set()

long_distance = IT2FS(distance_from_city_center_universe, trapezoid_mf, [13, 17, 30, 31, 1.0], trapezoid_mf, [17, 25, 30, 31, 0.999])
long_distance.plot('Duża odległość')
long_distance.check_set()

# Funkcje przynależności dla wieku mieszkania:

young_age_apartment = IT2FS(age_universe, trapezoid_mf, [-1, 0, 3, 7, 1.0], trapezoid_mf, [-1, 0, 3, 5, 0.999])
young_age_apartment.plot('Młode mieszkanie')
young_age_apartment.check_set()

middle_age_apartment = IT2FS(age_universe, trapezoid_mf, [7, 9, 15, 22, 1.0], trapezoid_mf, [7, 10, 15, 20, 0.5])
middle_age_apartment.plot('Średniego wieku mieszkanie')
middle_age_apartment.check_set()

old_age_apartment = IT2FS(age_universe, trapezoid_mf, [20, 30, 50, 51, 1.0], trapezoid_mf, [22, 35, 50, 51, 0.999])
old_age_apartment.plot('Stare mieszkanie')
old_age_apartment.check_set()

# Funkcje przynależności dla ceny mieszkania:

small_price = IT2FS(price_universe, trapezoid_mf, [-1, 0, 350000, 700000, 1.0], trapezoid_mf, [-1, 0, 250000, 500000, 0.999])
small_price.plot('Mała wartość')
small_price.check_set()

medium_price = IT2FS(price_universe, trapezoid_mf, [500000, 750000, 900000, 1200000, 1.0], tri_mf, [700000, 825000, 1000000, 0.999])
medium_price.plot('Średnia wartość')
medium_price.check_set()

big_price = IT2FS(price_universe, trapezoid_mf, [1100000, 1750000, 3500000, 3500001, 1.0], trapezoid_mf, [1500000, 2500000, 3500000, 3500001, 0.999])
big_price.plot('Duża wartość')
big_price.check_set()

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
IT2FS_plot(small_apartment, medium_apartment, large_apartment, giant_apartment, legends=["Małe", "Średnie", "Duże", "Gigantyczne"], title='Metraż mieszkania')
IT2FS_plot(short_distance, medium_distance, long_distance, legends=["Mała", "Średnia", "Duża"], title='Odległość mieszkania od ścisłego centrum')
IT2FS_plot(young_age_apartment, middle_age_apartment, old_age_apartment, legends=["Młode", "Średniego wieku", "Stare"], title='Wiek mieszkania')
IT2FS_plot(small_price, medium_price, big_price, legends=["Mała", "Średnia", "Duża"], title='Cena mieszkania')

# ==============================
# Definiowanie sterownika rozmytego.
# ==============================

# Sterownik rozmyty typu "Mamdani" z metodą defuzyfikacji "Centroid".
controller = Mamdani(t_norm=min_t_norm, s_norm=max_s_norm, method="Centroid", algorithm="KM")

# Trzy wejścia rozmyte i jedno wyjście dla sterownika rozmytego.
controller.add_input_variable('Metraż mieszkania')
controller.add_input_variable('Odległość mieszkania od ścisłego centrum')
controller.add_input_variable('Wiek mieszkania')
controller.add_output_variable('Wartość mieszkania [zł]')

# ==============================
# Definiowanie reguł rozmytych.
# ==============================

controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', small_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])

controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', medium_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])

controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', small_price)])
controller.add_rule([('Metraż mieszkania', large_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])

controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', short_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', medium_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', young_age_apartment)], [('Wartość mieszkania [zł]', big_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', middle_age_apartment)], [('Wartość mieszkania [zł]', medium_price)])
controller.add_rule([('Metraż mieszkania', giant_apartment), ('Odległość mieszkania od ścisłego centrum', long_distance), ('Wiek mieszkania', old_age_apartment)], [('Wartość mieszkania [zł]', small_price)])

# ==============================
# Ocenianie systemu rozmytego dla konkretnych wartości.
# ==============================

# Przykładowe dane wejściowe (wejście ostre).
square_footage_values = np.arange(10, 260, 10)
distance_from_city_center_values = np.arange(0, 35, 5)
age_values = np.arange(0, 55, 5)
for sf_value in square_footage_values:
    for dfcc_value in distance_from_city_center_values:
        for a_value in age_values:
            it2out, tr = controller.evaluate({"Metraż mieszkania": sf_value, "Odległość mieszkania od ścisłego centrum": dfcc_value, "Wiek mieszkania": a_value})
            price_value = round(crisp(tr['Wartość mieszkania [zł]']), 0)
            print(f'Cena mieszkania dla {sf_value} metrów kwadratowych, {dfcc_value} kilometrów odległości od ścisłego centrum, {a_value} lat wieku mieszkania: {price_value} zł')
        if dfcc_value != 30: print()
    if sf_value != 250: print()

# Wyświetlenie wszystkich wykresów.
plt.show()

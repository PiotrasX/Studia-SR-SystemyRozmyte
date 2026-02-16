import matplotlib.pyplot as plt
import numpy as np
from pyit2fls import Mamdani, min_t_norm, max_s_norm, IT2FS, tri_mf, ltri_mf, rtri_mf, crisp, IT2FS_plot

# ==============================
# Definiowanie uniwersów.
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

# Funkcje przynależności dla jakości jedzenia:

rancid_food = IT2FS(food_universe, rtri_mf, [3.01, 0, 1.0], rtri_mf, [2, 0, 0.99])
rancid_food.plot(title='Zepsute jedzenie')
rancid_food.check_set()

medium_food = IT2FS(food_universe, tri_mf, [2, 5.1, 8, 1.0], tri_mf, [4, 5.1, 6, 0.99])
medium_food.plot(title='Średnie jedzenie')
medium_food.check_set()

delicious_food = IT2FS(food_universe, ltri_mf, [7, 8, 1.0], ltri_mf, [8, 9, 0.99])
delicious_food.plot(title='Pyszne jedzenie')
delicious_food.check_set()

# Funkcje przynależności dla jakości obsługi:

low_service = IT2FS(service_universe, rtri_mf, [3.01, 0, 1.0], rtri_mf, [2, 0, 0.99])
low_service.plot(title='Słaby serwis')
low_service.check_set()

medium_service = IT2FS(service_universe, tri_mf,  [2, 5, 8, 1.0], tri_mf, [4, 5, 6, 1.0])
medium_service.plot(title='Średni serwis')
medium_service.check_set()

high_service = IT2FS(service_universe, ltri_mf, [6, 8, 1.0], ltri_mf, [7, 9, 0.99])
high_service.plot(title='Dobry serwis')
high_service.check_set()

# Funkcje przynależności dla wysokości napiwku:

low_tip = IT2FS(tip_universe, rtri_mf, [6, 0, 1.0], rtri_mf, [3, 0, 0.99])
low_tip.plot(title='Mały napiwek')
low_tip.check_set()

medium_tip = IT2FS(tip_universe, tri_mf, [4, 9, 14, 1.0], tri_mf, [8, 9, 10, 0.99])
medium_tip.plot(title='Średni napiwek')
medium_tip.check_set()

generous_tip = IT2FS(tip_universe, ltri_mf, [12, 15, 1.0], ltri_mf, [15, 17, 0.99])
generous_tip.plot(title='Hojny napiwek')
generous_tip.check_set()

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
IT2FS_plot(rancid_food, medium_food, delicious_food, legends=["Zepsute", "Średnie", "Pyszne"], title='Jakość jedzenia')
IT2FS_plot(low_service, medium_service, high_service, legends=["Słaba", "Średnia", "Dobra"], title='Jakość obsługi')
IT2FS_plot(low_tip, medium_tip, generous_tip, legends=["Mały", "Średni", "Hojny"], title='Wysokość napiwku [%]')

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
# Definiowanie reguł rozmytych.
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
food_values = np.arange(0, 11, 1)
service_values = np.arange(0, 11, 1)
for food_value in food_values:
    for service_value in service_values:
        it2out, tr = controller.evaluate({"Jakość jedzenia": food_value, "Jakość obsługi": service_value})
        tip_value = round(crisp(tr['Wysokość napiwku [%]']), 2)
        print(f'Napiwek w % dla jakości jedzenia {food_value} i jakości obsługi {service_value}: {tip_value} %')
    if food_value != 10: print()

# Wyświetlenie wszystkich wykresów.
plt.show()

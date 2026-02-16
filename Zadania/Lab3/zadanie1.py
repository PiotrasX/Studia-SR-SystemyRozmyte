import matplotlib.pyplot as plt
import numpy as np
from pyit2fls import T1TSK, T1FS, tri_mf, trapezoid_mf, T1FS_plot

# ==============================
# Definiowanie uniwersów.
# ==============================

food_universe = np.linspace(0.0, 10.0, 1000)
service_universe = np.linspace(0.0, 10.0, 1000)

# ==============================
# Definiowanie zbiorów rozmytych typu I (T1FS).
# ==============================

# Zbiory rozmyte dla jakości jedzenia:

rancid_food = T1FS(food_universe, trapezoid_mf, [-1, 0, 3, 5, 1.0])
rancid_food.plot('Zepsute jedzenie')

medium_food = T1FS(food_universe, tri_mf, [3, 5, 7, 1.0])
medium_food.plot('Średnie jedzenie')

delicious_food = T1FS(food_universe, trapezoid_mf, [6, 7, 10, 11, 1.0])
delicious_food.plot('Pyszne jedzenie')

# Zbiory rozmyte dla jakości obsługi:

low_service = T1FS(service_universe, trapezoid_mf, [-1, 0, 3, 5, 1.0])
low_service.plot('Słaby serwis')

medium_service = T1FS(service_universe, tri_mf, [3, 5, 7, 1.0])
medium_service.plot('Średni serwis')

high_service = T1FS(service_universe, trapezoid_mf, [6, 7, 10, 11, 1.0])
high_service.plot('Dobry serwis')

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
T1FS_plot(rancid_food, medium_food, delicious_food, legends=["Zepsute", "Średnie", "Pyszne"], title='Jakość jedzenia')
T1FS_plot(low_service, medium_service, high_service, legends=["Słaby", "Średni", "Dobry"], title='Jakość obsługi')

# ==============================
# Definiowanie funkcji wyjściowych (konsekwentów reguł TSK).
# ==============================

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
# Definiowanie reguł rozmytych TSK.
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

# Przykład a):
food_values = [0.0, 5.0, 10.0]
service_values = [0.0, 5.0, 10.0]
for food_value in food_values:
    for service_value in service_values:
        it2out = controller.evaluate({"Jakość jedzenia": food_value, "Jakość obsługi": service_value}, (food_value, service_value))
        print(f'Napiwek w % dla jakości jedzenia {food_value} i jakości obsługi {service_value}: {it2out['Wysokość napiwku [%]']} %')

# Przykład b):
food_value = 6.5
service_value = 5.0
it2out = controller.evaluate({"Jakość jedzenia": food_value, "Jakość obsługi": service_value}, (food_value, service_value))
print(f'\nNapiwek w % dla jakości jedzenia {food_value} i jakości obsługi {service_value}: {it2out['Wysokość napiwku [%]']} %')

# Wyświetlenie wszystkich wykresów.
plt.show()

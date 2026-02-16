import math
import matplotlib.pyplot as plt
import numpy as np
from pyit2fls import T1TSK, T1FS, tri_mf, trapezoid_mf, T1FS_plot

# ==============================
# Definiowanie uniwersów.
# ==============================

# Uniwersum wyprodukowanych sztuk produktów – skala od 0 do 150 produktów.
quantity_universe = np.linspace(0.0, 150.0, 1500)

# ==============================
# Definiowanie zbiorów rozmytych typu I (T1FS).
# ==============================

# Zbiory rozmyte dla wyprodukowanych sztuk produktów:

none_quantity = T1FS(quantity_universe, tri_mf, [-1, 0, 0.001, 1.0])
none_quantity.plot('Żadna produkcja')

normal_quantity = T1FS(quantity_universe, trapezoid_mf, [0.001, 0.002, 40, 50, 1.0])
normal_quantity.plot('Normalna produkcja')

slightly_above_average_quantity = T1FS(quantity_universe, tri_mf, [50, 80, 110, 1.0])
slightly_above_average_quantity.plot('Troszkę powyżej średniej produkcja')

much_above_average_quantity = T1FS(quantity_universe, trapezoid_mf, [90, 120, 150, 151, 1.0])
much_above_average_quantity.plot('Dużo powyżej średniej produkcja')

# Rysowanie wykresów funkcji przynależności dla każdego z uniwersum osobno.
T1FS_plot(none_quantity,
          normal_quantity,
          slightly_above_average_quantity,
          much_above_average_quantity,
          legends=["Żadna produkcja",
                   "Normalna produkcja",
                   "Troszkę powyżej średniej produkcja",
                   "Dużo powyżej średniej produkcja"],
          title='Ilość wyprodukowanych sztuk przedmiotu')

# ==============================
# Definiowanie funkcji wyjściowych (konsekwentów reguł TSK).
# ==============================

# Funkcja wyjściowa dla ujemnej premii.
def minus_bonus(y):
    return 0 * y - 20

# Funkcja wyjściowa dla braku premii.
def no_bonus(y):
    return 0 * y

# Funkcja wyjściowa dla małej premii.
def little_bonus(y):
    return (math.floor(y) - 50) * 5

# Funkcja wyjściowa dla dużej premii.
def big_bonus(y):
    return (math.floor(y) - 50) * 10

# ==============================
# Definiowanie sterownika rozmytego TSK.
# ==============================

# Sterownik rozmyty typu "TSK".
controller = T1TSK()

# Jedno wejście rozmyte i jedno wyjście dla sterownika rozmytego.
controller.add_input_variable('Ilość wyprodukowanych sztuk przedmiotu')
controller.add_output_variable('Wysokość premii [zł]')

# ==============================
# Definiowanie reguł rozmytych TSK.
# ==============================

controller.add_rule([('Ilość wyprodukowanych sztuk przedmiotu', none_quantity)], [('Wysokość premii [zł]', minus_bonus)])
controller.add_rule([('Ilość wyprodukowanych sztuk przedmiotu', normal_quantity)], [('Wysokość premii [zł]', no_bonus)])
controller.add_rule([('Ilość wyprodukowanych sztuk przedmiotu', slightly_above_average_quantity)], [('Wysokość premii [zł]', little_bonus)])
controller.add_rule([('Ilość wyprodukowanych sztuk przedmiotu', much_above_average_quantity)], [('Wysokość premii [zł]', big_bonus)])

# ==============================
# Ocenianie systemu rozmytego TSK dla konkretnych wartości.
# ==============================

# Przykładowe dane wejściowe (wejście ostre).
quantity_values = np.arange(0, 151, 1)
for quantity_value in quantity_values:
    it2out = controller.evaluate({"Ilość wyprodukowanych sztuk przedmiotu": quantity_value}, (quantity_value,))
    bonus = round(it2out['Wysokość premii [zł]'], 1)
    print(f'Wysokość premii dla {quantity_value} wyprodukowanych sztuk przedmiotów: {bonus} zł')

# Wyświetlenie wszystkich wykresów.
plt.show()

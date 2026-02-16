from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from FuzzyKNN.fuzzy_knn import FuzzyKNN

# ==================================================
# Ten program porównuje klasyczny ostry algorytm kNN z jego rozmytą wersją Fuzzy kNN.
#
# Celem programu jest zbadanie różnic między klasycznym kNN a jego wariantem,
# który zamiast jednoznacznych etykiet przypisuje stopnie przynależności do klas.
# ==================================================

# --------------------------------------------------
# Wczytanie danych z pliku CSV.
# --------------------------------------------------

# Dane zawierają informacje o wzroście, wadze i etykiecie decyzyjnej:
# 	- wzrost w centymetrach,
# 	- waga w kilogramach,
# 	- wartość BMI (1 - prawidłowe BMI / 0 - nieprawidłowe BMI).
data = pd.read_csv('bmi.csv')

# Wyświetlanie przykładowych danych, by sprawdzić ich strukturę.
print(f'Dane wejściowe:\n{data}\n')

# --------------------------------------------------
# Dane wejściowe (atrybuty) i wartości etykiet decyzyjnych.
# --------------------------------------------------

# Macierz atrybutów warunkowych X (height, weight).
X = data.iloc[:, :-1].to_numpy()

# Wektor atrybutu decyzyjnego y (bmi_ok).
y = data.iloc[:, -1].to_numpy()

# --------------------------------------------------
# Wymiary danych.
# --------------------------------------------------

# Warto upewnić się, czy liczba próbek w X i y się zgadza.
# Zawsze musi być zgodność wymiarów pomiędzy X a y.
print(f'Wymiar danych wejściowych X: {X.shape}')  # Macierz.
print(f'Wymiar etykiet decyzyjnych y: {y.shape}\n')  # Wektor jednowymiarowy.

# --------------------------------------------------
# Podział danych na zbiór treningowy i testowy.
# --------------------------------------------------

# Funkcja train_test_split losowo dzieli dane:
# 	- 70% próbek na trening (do nauki modelu),
# 	- 30% próbek na test (do sprawdzenia jakości).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# --------------------------------------------------
# Uczenie klasycznego kNN.
# --------------------------------------------------

# Tworzenie obiektu klasyfikatora z 3 sąsiadami. Oznacza to, że dla każdej nowej próbki
# algorytm patrzy na 3 najbliższe punkty w danych treningowych i wybiera klasę większościową.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --------------------------------------------------
# Przewidywanie etykiet dla danych testowych.
# --------------------------------------------------

classic_y_predicted = knn.predict(X_test)  # Wyznaczanie decyzji dla danych testowych.
print('Klasyczny kNN:')
print(f'\tEtykiety przypisane przez klasyczny klasyfikator: {classic_y_predicted}')
print(f'\tRzeczywiste etykiety: {y_test}')
print(f'\tKlasa o najwyższym prawdopodobieństwie z predict_proba dla każdej próbki testowej: {np.argmax(knn.predict_proba(X_test), axis=1)}')
print(f'\tMacierz prawdopodobieństw:\n{knn.predict_proba(X_test)}\n')

# --------------------------------------------------
# Uczenie rozmytego kNN.
# --------------------------------------------------

# Fuzzy kNN zamiast przypisywać konkretną klasę (0 lub 1),
# określa stopień przynależności próbki do każdej z klas.
fuzzy_knn = FuzzyKNN(k=3)
fuzzy_knn.fit(X_train, y_train)

# --------------------------------------------------
# Przewidywanie etykiet dla danych testowych.
# --------------------------------------------------

fuzzy_y_predicted = fuzzy_knn.predict(X_test)  # Wyznaczanie decyzji dla danych testowych.
print('Rozmyty Fuzzy kNN:')
print(f'\tEtykiety przypisane przez rozmyty klasyfikator: {fuzzy_y_predicted}')
print(f'\tRzeczywiste etykiety: {y_test}')
print(f'\tKlasa o najwyższym prawdopodobieństwie z predict_proba dla każdej próbki testowej: {np.argmax(fuzzy_knn.predict_proba(X_test), axis=1)}')
print(f'\tMacierz prawdopodobieństw:\n{fuzzy_knn.predict_proba(X_test)}\n')

# --------------------------------------------------
# Ocena jakości i skuteczności obu klasyfikatorów (modeli).
# --------------------------------------------------

print(f'Etykiety przypisane przez klasycznym kNN: {knn.predict(X_test)}')
print(f'Etykiety przypisane przez rozmytym Fuzzy kNN: {fuzzy_knn.predict(X_test)}\n')
print(f'Dokładność klasycznego kNN: {knn.score(X_test, y_test)}')
print(f'Dokładność rozmytego Fuzzy kNN: {fuzzy_knn.score(X_test, y_test)}\n')

# --------------------------------------------------
# Analiza błędów poprzez macierz pomyłek.
# --------------------------------------------------

# Macierz pomyłek (confusion matrix) pokazuje, ile próbek z każdej klasy
# zostało sklasyfikowanych poprawnie (na przekątnej) i błędnie (poza nią).
classic_cm = confusion_matrix(y_test, classic_y_predicted)
fuzzy_cm = confusion_matrix(y_test, fuzzy_y_predicted)
print(f'Macierz pomyłek dla klasycznego kNN:\n{classic_cm}\n')
print(f'Macierz pomyłek dla rozmytego Fuzzy kNN:\n{fuzzy_cm}\n')

# Wizualizacja macierzy pomyłek klasycznego kNN w formie graficznej.
disp = ConfusionMatrixDisplay(confusion_matrix=classic_cm, display_labels=knn.classes_)
disp.plot()
plt.title('Macierz pomyłek klasycznego kNN')
plt.xlabel('Etykiety przewidziane przez klasyfikator')
plt.ylabel('Etykiety rzeczywiste (prawidłowe)')

# Wizualizacja macierzy pomyłek rozmytego Fuzzy kNN w formie graficznej.
disp2 = ConfusionMatrixDisplay(confusion_matrix=fuzzy_cm, display_labels=fuzzy_knn.classes)
disp2.plot()
plt.title('Macierz pomyłek rozmytego Fuzzy kNN')
plt.xlabel('Etykiety przewidziane przez klasyfikator')
plt.ylabel('Etykiety rzeczywiste (prawidłowe)')

plt.show()

# --------------------------------------------------
# Kroswalidacja (Cross-Validation).
# --------------------------------------------------

knn_pipeline = Pipeline([
	('scaler', StandardScaler()),
	('knn', KNeighborsClassifier(n_neighbors=3))
])
fuzzy_knn_pipeline = Pipeline([
	('scaler', StandardScaler()),
	('fuzzy_knn', FuzzyKNN(k=3))
])

# Wyniki klasycznego kNN dla miary 'accuracy' oraz 'roc_auc'.
classic_knn_acc_cross_val = cross_val_score(cv=5, estimator=knn_pipeline, X=X, y=y, scoring='accuracy')
classic_knn_roc_auc_cross_val = cross_val_score(cv=5, estimator=knn_pipeline, X=X, y=y, scoring='roc_auc')
print('Dla klasycznego kNN:')
print(f'\tAccuracy z kroswalidacji: {classic_knn_acc_cross_val}')
print(f'\tRoc auc z kroswalidacji: {classic_knn_roc_auc_cross_val}')
print(f'\tŚrednie accuracy: {np.mean(classic_knn_acc_cross_val):.7f} +- {stdev(classic_knn_acc_cross_val):.7f} (odchylenie standardowe)')
print(f'\tŚrednie roc auc: {np.mean(classic_knn_roc_auc_cross_val):.7f} +- {stdev(classic_knn_roc_auc_cross_val):.7f} (odchylenie standardowe)\n')

# Wyniki rozmytego Fuzzy kNN dla miary 'accuracy' oraz 'roc_auc'.
fuzzy_knn_acc_cross_val = cross_val_score(cv=5, estimator=fuzzy_knn_pipeline, X=X, y=y, scoring='accuracy')
fuzzy_knn_roc_auc_cross_val = cross_val_score(cv=5, estimator=fuzzy_knn_pipeline, X=X, y=y, scoring='roc_auc')
print('Dla rozmytego Fuzzy kNN:')
print(f'\tAccuracy z kroswalidacji: {classic_knn_acc_cross_val}')
print(f'\tRoc auc z kroswalidacji: {fuzzy_knn_roc_auc_cross_val}')
print(f'\tŚrednie accuracy: {np.mean(classic_knn_acc_cross_val):.7f} +- {stdev(classic_knn_acc_cross_val):.7f} (odchylenie standardowe)')
print(f'\tŚrednie roc auc: {np.mean(fuzzy_knn_roc_auc_cross_val):.7f} +- {stdev(fuzzy_knn_roc_auc_cross_val):.7f} (odchylenie standardowe)')

# --------------------------------------------------
# Wizualizacja danych na wykresie.
# --------------------------------------------------

# Kolorami oznaczone są klasy BMI (0 = nieprawidłowe, 1 = prawidłowe).
# Można zobaczyć, że granica pomiędzy klasami nie jest liniowa.
myc_map  = ListedColormap([[0.267004, 0.004874, 0.329415, 1.0],
						   [0.993248, 0.906157, 0.143936, 1.0]])

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Etykiety rzeczywiste (prawidłowe)')
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=knn.predict(X_test), cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Klasyczny kNN')
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict(X_test), cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Zbinaryzowany rozmyty Fuzzy kNN')
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.suptitle('Porównanie klasycznego kNN i rozmytego Fuzzy kNN (etykiety decyzyjne)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict_proba(X_test)[:, 0])
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Rozmyty Fuzzy kNN – przynależność do klasy 0')
plt.colorbar(label='Stopień przynależności')
plt.clim(0, 1)

plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict_proba(X_test)[:, 1])
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Rozmyty Fuzzy kNN – przynależność do klasy 1')
plt.colorbar(label='Stopień przynależności')
plt.clim(0, 1)

plt.suptitle('Rozmyta przynależność do klas w rozmytym Fuzzy kNN')
plt.tight_layout()
plt.show()

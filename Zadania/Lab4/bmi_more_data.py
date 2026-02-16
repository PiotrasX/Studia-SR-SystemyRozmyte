from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# ==================================================
# Ten program to klasyczny algorytm kNN dla danych BMI w wersji rozszerzonej.
#
# Celem programu jest pokazanie działania klasyfikatora kNN (k-Nearest Neighbors) na prostych danych
# o wzroście i wadze człowieka. Dodatkowo wyniki będą porównane z metodami klasteryzacji (DBSCAN, K-Means).
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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

y_predicted = knn.predict(X_test)  # Wyznaczanie decyzji dla danych testowych.
print(f'Etykiety przypisane przez klasyfikator: {y_predicted}')
print(f'Rzeczywiste etykiety: {y_test}\n')

# --------------------------------------------------
# Ocena skuteczności klasyfikatora.
# --------------------------------------------------

# TN (true negatives), FP (false positives), FN (false negatives), TP (true positives).
tn, fp, fn, tp = confusion_matrix(y_test, knn.predict(X_test)).ravel()

# Wskaźniki skuteczności.
tpr = tp / (tp + fn)  # True Positive Rate (czułość).
fpr = fp / (fp + tn)  # False Positive Rate (odsetek fałszywych alarmów).

print(f'Dokładność kNN (score) = {knn.score(X_test, y_test)}')
print(f'TPR (True Positive Rate): {tpr:.3f}')
print(f'FPR (False Positive Rate): {fpr:.3f}\n')

# --------------------------------------------------
# Analiza błędów poprzez macierz pomyłek.
# --------------------------------------------------

# Macierz pomyłek (confusion matrix) pokazuje, ile próbek z każdej klasy
# zostało sklasyfikowanych poprawnie (na przekątnej) i błędnie (poza nią).
cm = confusion_matrix(y_test, y_predicted)
print(f'Macierz pomyłek:\n{cm}\n')

# Wizualizacja macierzy pomyłek w formie graficznej.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.title('Macierz pomyłek klasyfikatora kNN')
plt.xlabel('Etykiety przewidziane przez klasyfikator')
plt.ylabel('Etykiety rzeczywiste (prawidłowe)')
plt.show()

# --------------------------------------------------
# Kroswalidacja (Cross-Validation).
# --------------------------------------------------

# Kroswalidacja pozwala ocenić stabilność modelu.
# Dane są dzielone na 5 różnych części (fold'ów),
# a model jest testowany 5 razy na różnych podziałach danych.
skf = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)

# Wyniki dla miary 'accuracy'.
acc_cross_val = cross_val_score(cv=skf, estimator=knn, X=X, y=y, scoring='accuracy')

# Wyniki dla miary 'roc_auc' (lepiej pokazuje jakość modelu).
roc_auc_cross_val = cross_val_score(cv=skf, estimator=knn, X=X, y=y, scoring='roc_auc')

# Wypisanie wyników.
print(f'Accuracy z kroswalidacji: {acc_cross_val}')
print(f'Roc auc z kroswalidacji: {roc_auc_cross_val}\n')

# Wypisanie średnich wyników z pięciu podziałów z odchyleniem standardowym.
print(f'Średnie accuracy: {np.mean(acc_cross_val):.7f} +- {stdev(acc_cross_val):.7f} (odchylenie standardowe)')
print(f'Średnie roc auc: {np.mean(roc_auc_cross_val):.7f} +- {stdev(roc_auc_cross_val):.7f} (odchylenie standardowe)')

# --------------------------------------------------
# Wizualizacja danych na wykresie.
# --------------------------------------------------

# Kolorami oznaczone są klasy BMI (0 = nieprawidłowe, 1 = prawidłowe).
# Można zobaczyć, że granica pomiędzy klasami nie jest liniowa.
myc_map  = ListedColormap([[0.267004, 0.004874, 0.329415, 1.0],
						   [0.993248, 0.906157, 0.143936, 1.0]])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Wizualizacja danych BMI')
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Etykiety rzeczywiste (prawidłowe)')
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted, cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title('Etykiety przewidziane przez klasyfikator')
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.suptitle('Porównanie decyzji klasyfikatora z prawdziwymi etykietami')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Klasteryzacja (porównanie z metodami bez nadzoru).
# --------------------------------------------------

# Celem tej sekcji jest pokazanie, że algorytmy klasteryzacji (DBSCAN, K-Means)
# nie zawsze odwzorowują klasy decyzyjne, jeśli nie mają etykiet do nauki.

plt.figure(figsize=(10, 8))

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title("Etykiety rzeczywiste (prawidłowe)")
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=DBSCAN(min_samples=10, eps=12).fit_predict(X), cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title("Etykiety przypisane przez DBSCAN")
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.subplot(223)
plt.scatter(X[:, 0], X[:, 1], c=KMeans(n_clusters=3, n_init=10).fit_predict(X), cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title("Etykiety przypisane przez K-Means")
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.subplot(224)
plt.scatter(X_test[:, 0], X_test[:, 1], c=knn.predict(X_test), cmap=myc_map)
plt.xlabel('Wzrost [cm]')
plt.ylabel('Waga [kg]')
plt.title("Etykiety przypisane przez k-NN")
plt.colorbar(label='Nieprawidłowe BMI (0) / Prawidłowe BMI (1)')
plt.clim(0, 1)

plt.suptitle("Porównanie metod klasteryzacji i klasyfikacji")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Interpretacja wyników.
# --------------------------------------------------

# Algorytm k-NN przypisuje etykiety na podstawie wzorca (uczenia z nadzorem).
# Algorytm DBSCAN szuka gęstych skupisk punktów, ignoruje etykiety.
# Algorytm K-Means dzieli dane na k grup o minimalnym rozrzucie wewnętrznym.

# Wyniki pokazują, że k-NN lepiej odtwarza prawdziwe klasy,
# ponieważ został wytrenowany na danych z etykietami (y),
# a metody klasteryzacji działają "w ciemno".

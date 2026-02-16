from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# ==================================================
# Ten program to klasyczny algorytm kNN dla danych BMI.
#
# Celem programu jest pokazanie działania klasyfikatora kNN (k-Nearest Neighbors) na prostych danych
# o wzroście i wadze człowieka. Na tej podstawie klasyfikator ocenia, czy dana osoba ma prawidłowe BMI.
# ==================================================

# --------------------------------------------------
# Dane wejściowe (atrybuty).
# --------------------------------------------------

# Każda lista to jedna próbka (osoba), a cechy (dane) zawarte w niej to:
# 	- wzrost w centymetrach,
# 	- waga w kilogramach.
# W ten sposób powstaje macierz atrybutów X (16 próbek × 2 cechy).
X = np.array([[152, 45], [152, 55], [152, 65], [152, 75], [152, 90], [170, 45], [170, 55], [170, 65],
			  [170, 75], [170, 90], [190, 45], [190, 55], [190, 65], [190, 75], [190, 90], [190, 120]])

# --------------------------------------------------
# Wartości etykiet decyzyjnych.
# --------------------------------------------------

# Dla każdej osoby określa się, czy BMI jest prawidłowe (1) czy nie (0).
# Wartości odpowiadają kolejno próbom z macierzy X.
# To jest wektor klas (etykiet), który uczy model przewidywać.
y = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0])

# --------------------------------------------------
# Wymiary danych.
# --------------------------------------------------

# Warto upewnić się, czy liczba próbek w X i y się zgadza.
# Zawsze musi być zgodność wymiarów pomiędzy X a y.
print(f'Wymiar danych wejściowych X: {X.shape}')  # 16 wierszy × 2 kolumny.
print(f'Wymiar etykiet decyzyjnych y: {y.shape}\n')  # 16 etykiet (wektor jednowymiarowy).

# --------------------------------------------------
# Podział danych na zbiór treningowy i testowy.
# --------------------------------------------------

# Funkcja train_test_split losowo dzieli dane:
# 	- 85% próbek na trening (do nauki modelu),
# 	- 15% próbek na test (do sprawdzenia jakości).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# --------------------------------------------------
# Uczenie klasycznego kNN.
# --------------------------------------------------

# Tworzenie obiektu klasyfikatora z 3 sąsiadami. Oznacza to, że dla każdej nowej próbki
# algorytm patrzy na 3 najbliższe punkty w danych treningowych i wybiera klasę większościową.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)  # Uczenie modelu.

# --------------------------------------------------
# Przewidywanie etykiet dla danych testowych.
# --------------------------------------------------

y_predicted = knn.predict(X_test)  # Wyznaczanie decyzji dla danych testowych.
print(f'Etykiety przypisane przez klasyfikator: {y_predicted}')
print(f'Rzeczywiste etykiety: {y_test}\n')

# --------------------------------------------------
# Ocena skuteczności klasyfikatora.
# --------------------------------------------------

print(f'Dokładność kNN (score) = {knn.score(X_test, y_test)}')
print(f'Dokładność kNN (accuracy_score) = {accuracy_score(y_test, y_predicted)}\n')

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

# Kroswalidacja (StratifiedKFold) pozwala ocenić stabilność modelu.
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

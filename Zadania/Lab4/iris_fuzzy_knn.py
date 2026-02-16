from statistics import stdev

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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
# Wczytanie danych.
# --------------------------------------------------

X, y = load_iris(return_X_y=True)

# Ładowanie nazw atrybutów warunkowych.
attributes_names = load_iris().feature_names
decision_names = load_iris().target_names

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

# Wyniki klasycznego kNN dla miary 'roc_auc_ovo'.
classic_knn_acc_cross_val = cross_val_score(cv=5, estimator=knn_pipeline, X=X, y=y, scoring='accuracy')
classic_knn_roc_auc_cross_val = cross_val_score(cv=5, estimator=knn_pipeline, X=X, y=y, scoring='roc_auc_ovo')
print('Dla klasycznego kNN:')
print(f'\tAccuracy z kroswalidacji: {classic_knn_acc_cross_val}')
print(f'\tRoc auc ovo z kroswalidacji: {classic_knn_roc_auc_cross_val}')
print(f'\tŚrednie accuracy: {np.mean(classic_knn_acc_cross_val):.7f} +- {stdev(classic_knn_acc_cross_val):.7f} (odchylenie standardowe)')
print(f'\tŚrednie roc auc ovo: {np.mean(classic_knn_roc_auc_cross_val):.7f} +- {stdev(classic_knn_roc_auc_cross_val):.7f} (odchylenie standardowe)\n')

# Wyniki rozmytego Fuzzy kNN dla miary 'roc_auc_ovo'.
fuzzy_knn_acc_cross_val = cross_val_score(cv=5, estimator=fuzzy_knn_pipeline, X=X, y=y, scoring='accuracy')
fuzzy_knn_roc_auc_cross_val = cross_val_score(cv=5, estimator=fuzzy_knn_pipeline, X=X, y=y, scoring='roc_auc_ovo')
print('Dla rozmytego Fuzzy kNN:')
print(f'\tAccuracy z kroswalidacji: {classic_knn_acc_cross_val}')
print(f'\tRoc auc ovo z kroswalidacji: {fuzzy_knn_roc_auc_cross_val}')
print(f'\tŚrednie accuracy: {np.mean(classic_knn_acc_cross_val):.7f} +- {stdev(classic_knn_acc_cross_val):.7f} (odchylenie standardowe)')
print(f'\tŚrednie roc auc ovo: {np.mean(fuzzy_knn_roc_auc_cross_val):.7f} +- {stdev(fuzzy_knn_roc_auc_cross_val):.7f} (odchylenie standardowe)')

# --------------------------------------------------
# Wizualizacja danych na wykresie.
# --------------------------------------------------

# Kolorami oznaczone są klasy BMI (0 = nieprawidłowe, 1 = prawidłowe).
# Można zobaczyć, że granica pomiędzy klasami nie jest liniowa.
myc_map  = ListedColormap([[0.267004, 0.004874, 0.329415, 1.0],
						   [0.128729, 0.563265, 0.551229],
						   [0.993248, 0.906157, 0.143936, 1.0]])

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Etykiety rzeczywiste')
plt.clim(0, 2)

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=knn.predict(X_test), cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Klasyczny kNN')
plt.clim(0, 2)

plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict(X_test), cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Zbinaryzowany rozmyty Fuzzy kNN')
plt.clim(0, 2)

plt.suptitle('Porównanie klasycznego kNN i rozmytego Fuzzy kNN')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict_proba(X_test)[:, 0])
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Rozmyty Fuzzy kNN – przynależność do klasy 0')
plt.clim(0, 1)

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict_proba(X_test)[:, 1])
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Rozmyty Fuzzy kNN – przynależność do klasy 1')
plt.clim(0, 1)

plt.subplot(1, 3, 3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=fuzzy_knn.predict_proba(X_test)[:, 2])
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Rozmyty Fuzzy kNN – przynależność do klasy 2')
plt.clim(0, 1)

plt.suptitle('Rozmyta przynależność do klas w rozmytym Fuzzy kNN')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Wizualizacja 3D przynależności do poszczególnych klastrów.
# --------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=fuzzy_knn.predict(X_test), cmap=myc_map)
ax.set_xlabel(attributes_names[0])
ax.set_ylabel(attributes_names[1])
# noinspection PyUnresolvedReferences
ax.set_zlabel(attributes_names[2])
plt.title('Zbinaryzowany rozmyty Fuzzy kNN')
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Przewidziana klasa (0, 1, 2)')
plt.suptitle('Przynależność 3D próbek testowych w klasyfikatorze rozmytym Fuzzy kNN')
plt.show()

import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# ==================================================
# Ten program demonstruje działanie algorytmu Fuzzy C-Means (FCM)
# na zbiorze danych IRIS oraz porównuje wyniki z klasycznym algorytmem K-Means.
#
# FCM to rozmyty algorytm klasteryzacji, w którym każda próbka
# może należeć do kilku klastrów jednocześnie, z różnym stopniem przynależności.
# ==================================================

# --------------------------------------------------
# Wczytanie danych.
# --------------------------------------------------

X, y = load_iris(return_X_y=True)

# Ładowanie nazw atrybutów warunkowych.
attributes_names = load_iris().feature_names
decision_names = load_iris().target_names

print(f'Dane wejściowe:\n{fuzz.cluster.cmeans(data=X.T, c=3, m=2, error=0.001, maxiter=10000)}')

# --------------------------------------------------
# Uruchomienie algorytmu Fuzzy C-Means.
# --------------------------------------------------

# Implementacja Fuzzy C-Means zwraca odpowiednio:
# 	- środki klastrów,
# 	- początkowy podział na klastry,
# 	- końcowy podział na klastry,
# 	- macierz odległości Euklidesa pomiędzy środkiem klastra a obiektami,
# 	- historię funkcji celu,
# 	- wykonaną liczbę iteracji,
# 	- obliczony FPC (Fuzzy Partition Coefficient).

# Parametry Fuzzy C-Means:
# 	- data: dane wejściowe (transponowane, ponieważ biblioteka skfuzzy wymaga kolumn = próbek),
# 	- c: liczba klastrów,
# 	- m: współczynnik rozmycia (m > 1),
# 	- error: próg błędu zatrzymania,
# 	- maxiter: maksymalna liczba iteracji.
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data=X.T, c=3, m=2, error=0.001, maxiter=10000)

print(f'\nŚrodki klastrów (cluster centers):\n{cntr}')
print(f'\nMacierz przynależności U (membership matrix):\n{u.T}')
print(f'\nHistoria funkcji celu:\n{jm}')
print(f'\nLiczba wykonanych iteracji:\n{p}')
print(f'\nFPC (Fuzzy Partition Coefficient):\n{fpc}\n')

# --------------------------------------------------
# Wizualizacja danych na wykresie.
# --------------------------------------------------

myc_map = ListedColormap([[0.267004, 0.004874, 0.329415, 1.0],
                         [0.128729, 0.563265, 0.551229],
                         [0.993248, 0.906157, 0.143936, 1.0]])

# --------------------------------------------------
# Wizualizacja i porównanie FCM z klasycznym K-Means na wybranych próbkach.
# --------------------------------------------------

# Dla przejrzystości wizualizacji wybrano po 10 obiektów z każdej klasy.
indexes = np.ravel(np.array([range(10), range(50,60), range(100,110)]))

plt.figure(figsize=(12, 6))

# Rzeczywiste etykiety.
plt.subplot(231)
plt.scatter(X[indexes, 0], X[indexes, 1], c=y[indexes], cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Etykiety rzeczywiste')
plt.clim(0, 2)

# Zamiana rozmytej przynależności na ostrą (największe prawdopodobieństwo).
y_crisp = np.argmax(u.T[indexes], axis=1)
print(f'Przynależność prób FCM (po binaryzacji):\n{y_crisp}')

# Obliczenie klasycznego K-Means.
kmeans = KMeans(n_clusters=3, n_init=10).fit(X[indexes])
kmeans_cluster_centers = kmeans.cluster_centers_

# Klasyczny K-Means.
plt.subplot(232)
plt.scatter(kmeans_cluster_centers[:, 0], kmeans_cluster_centers[:, 1], marker='x', color='red')
plt.scatter(X[indexes, 0], X[indexes, 1], c=kmeans.predict(X[indexes]), cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Klasyczny K-Means')
plt.clim(0, 2)

# Zbinaryzowany Fuzzy C-Means.
plt.subplot(233)
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.scatter(X[indexes, 0], X[indexes, 1], c=y_crisp, cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Zbinaryzowany Fuzzy C-Means')
plt.clim(0, 2)

# Przynależność do klasy 0 w FCM.
plt.subplot(234)
plt.scatter(X[indexes, 0], X[indexes, 1], c=u.T[indexes, 0])
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Fuzzy C-Means – przynależność do klasy 0')
plt.clim(0, 1)

# Przynależność do klasy 1 w FCM.
plt.subplot(235)
plt.scatter(X[indexes, 0], X[indexes, 1], c=u.T[indexes, 1])
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Fuzzy C-Means – przynależność do klasy 1')
plt.clim(0, 1)

# Przynależność do klasy 2 w FCM.
plt.subplot(236)
plt.scatter(X[indexes, 0], X[indexes, 1], c=u.T[indexes, 2])
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Fuzzy C-Means – przynależność do klasy 2')
plt.clim(0, 1)

plt.suptitle('Porównanie algorytmów Fuzzy C-Means i K-Means (na wybranych danych)')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Wizualizacja i porównanie FCM z klasycznym K-Means na wszystkich próbkach.
# --------------------------------------------------

plt.figure(figsize=(12, 6))

# Rzeczywiste etykiety.
plt.subplot(231)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=myc_map)
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Etykiety rzeczywiste')
plt.clim(0, 2)

# Obliczenie klasycznego K-Means.
kmeans = KMeans(n_clusters=3).fit(X)
kmeans_cluster_centers = kmeans.cluster_centers_

# Klasyczny K-Means.
plt.subplot(232)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap=myc_map)
plt.scatter(kmeans_cluster_centers[:, 0], kmeans_cluster_centers[:, 1], marker='x', color='red')
plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap=myc_map)
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Klasyczny K-Means')
plt.clim(0, 2)

# Zbinaryzowany Fuzzy C-Means.
plt.subplot(233)
plt.scatter(X[:, 0], X[:, 1], c=np.argmax(u.T, axis=1), cmap=myc_map)
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Gatunek')
plt.title('Zbinaryzowany Fuzzy C-Means')
plt.clim(0, 2)

# Przynależność do klasy 0 w FCM.
plt.subplot(234)
plt.scatter(X[:, 0], X[:, 1], c=u.T[:, 0])
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Fuzzy C-Means – przynależność do klasy 0')
plt.clim(0, 1)

# Przynależność do klasy 1 w FCM.
plt.subplot(235)
plt.scatter(X[:, 0], X[:, 1], c=u.T[:, 1])
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Fuzzy C-Means – przynależność do klasy 1')
plt.clim(0, 1)

# Przynależność do klasy 2 w FCM.
plt.subplot(236)
plt.scatter(X[:, 0], X[:, 1], c=u.T[:, 2])
plt.scatter(cntr[:, 0], cntr[:, 1], marker='x', color='red')
plt.xlabel(attributes_names[0])
plt.ylabel(attributes_names[1])
plt.colorbar(label='Stopień przynależności')
plt.title('Fuzzy C-Means – przynależność do klasy 2')
plt.clim(0, 1)

plt.suptitle('Porównanie algorytmów Fuzzy C-Means i K-Means (na całym zbiorze danych)')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Wizualizacja 3D przynależności do poszczególnych klastrów.
# --------------------------------------------------

for i in range(3):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=u.T[:, i])
	ax.scatter(cntr[:, 0], cntr[:, 1], cntr[:, 2], marker='x', color='red')
	ax.set_xlabel(attributes_names[0])
	ax.set_ylabel(attributes_names[1])
	# noinspection PyUnresolvedReferences
	ax.set_zlabel(attributes_names[2])
	plt.title(f'FCM – przynależność do klasy {decision_names[i]}')
	cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
	cbar.set_label('Stopień przynależności')
	plt.suptitle('Przynależność 3D do klastrów w algorytmie Fuzzy C-Means')

plt.show()

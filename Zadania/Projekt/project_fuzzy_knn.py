from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from FuzzyKNN.fuzzy_knn import FuzzyKNN


# ===============================================================================
# Program porównuje klasyczny ostry algorytm kNN z jego rozmytą wersją Fuzzy kNN.
#
# Celem programu jest zbadanie różnic między klasycznym kNN a rozmytym Fuzzy kNN.
# Porównywane są dwa zbiory danych:
# 	- Seeds (Ziarna Pszenicy),
# 	- Banknote Authentication (Uwierzytelnianie Banknotów).
# ===============================================================================

# --------------------------------------------------
# Definiowanie funkcji pomocniczych.
# --------------------------------------------------

# Funkcja do wyświetlania tablicy NumPy w jednej linii.
def one_line(arr):
	return np.array2string(
		arr,
		separator=', ',
		max_line_width=1000
	)

# Funkcja do wyświetlania legendy wykresów w Plot.
def add_legend(axis, class_names, cmap):
	handles = []
	for idx, name in enumerate(class_names):
		handles.append(
			Line2D(
				[0], [0],
				marker='o',
				color='w',
				label=name,
				markerfacecolor=cmap(idx),
				markersize=8
			)
		)
	axis.legend(handles=handles, title="Klasy", loc="best")

# Funkcja wybierająca zbiór danych oraz wersję danych (skalowane / nieskalowane).
def get_data(dataset="Seeds", scaled=True):
	if dataset == "Seeds":
		if not scaled:
			return X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds
		return X_train_scaled_seeds, X_test_scaled_seeds, y_train_seeds, y_test_seeds
	if dataset == "Banknote Authentication":
		if not scaled:
			return X_train_banknotes, X_test_banknotes, y_train_banknotes, y_test_banknotes
		return X_train_scaled_banknotes, X_test_scaled_banknotes, y_train_banknotes, y_test_banknotes
	return None

# Funkcja przeprowadzająca proces uczenia i testowania klasyfikatora.
def test_classifier(classifier, dataset="Seeds", scaled=True):
	x_train, x_test, y_train, y_test = get_data(dataset, scaled)  # Pobranie danych.
	classifier.fit(x_train, y_train)  # Uczenie maszynowe.
	y_predicted = classifier.predict(x_test)  # Wyznaczanie decyzji dla danych testowych.
	accuracy = classifier.score(x_test, y_test)  # Wyznaczenie dokładności klasyfikatora.
	cm = confusion_matrix(y_test, y_predicted)  # Wyznaczenie macierzy pomyłek.
	return y_predicted, accuracy, cm

# Funkcja wypisująca wyniki klasyfikatora i kroswalidacji.
def print_result(accuracy, cm, cv_accuracy, cv_roc_auc):
	print(f'\t\tDokładność klasyfikatora: {accuracy:.5f}')
	print(f'\t\tMacierz pomyłek klasyfikatora:\n{cm}')
	print(f'\t\tAccuracy z kroswalidacji: {cv_accuracy}')
	print(f'\t\tROC AUC z kroswalidacji: {cv_roc_auc}')
	print(f'\t\tŚrednie accuracy z kroswalidacji: {np.mean(cv_accuracy):.5f} +- {stdev(cv_accuracy):.5f} (odchylenie standardowe)')
	print(f'\t\tŚrednie ROC AUC z kroswalidacji: {np.mean(cv_roc_auc):.5f} +- {stdev(cv_roc_auc):.5f} (odchylenie standardowe)\n')

# Funkcja wypisująca najlepszy lub najgorszy klasyfikator.
def print_best_or_worst_classifier(title, accuracy, cm):
	print(f'\t\tKlasyfikator {title}')
	print(f'\t\tDokładność klasyfikatora: {accuracy:.5f}')
	print(f'\t\tMacierz pomyłek klasyfikatora:\n{cm}\n')

# --------------------------------------------------
# Wczytanie danych: Seeds.
# --------------------------------------------------

# Zbiór danych: Seeds (Ziarna Pszenicy).
# Struktura: 7 cech numerycznych + klasa (1 -> Kama, 2 -> Rosa, 3 -> Canadian).

data_seeds = np.loadtxt("data_seeds.txt")
y_seeds = data_seeds[:, -1].astype(int) - 1
X_seeds = data_seeds[:, :-1]
# X_seeds = X_seeds[:, [0, 1, 3, 4]]  # Tutaj pierwsza, druga, czwarta i piąta cecha.

feature_names_seeds = [
	"Pole powierzchni",
	"Obwód",
	"Zwartość",
	"Długość ziarna",
	"Szerokość ziarna",
	"Współczynnik asymetrii",
	"Długość bruzdy"
]
# feature_names_seeds = [feature_names_seeds[i] for i in [0, 1, 3, 4]]
class_names_seeds = ["Kama", "Rosa", "Canadian"]

print("Zbiór danych: Seeds (Ziarna Pszenicy)")
print(f"Liczba próbek: {X_seeds.shape[0]}")
print(f"Liczba cech: {X_seeds.shape[1]}")
print(f"Klasy: {class_names_seeds}\n")

# --------------------------------------------------
# Wczytanie danych: Banknote Authentication.
# --------------------------------------------------

# Zbiór danych: Banknote Authentication (Uwierzytelnianie Banknotów).
# Struktura: 4 cechy numeryczne + klasa (0 -> Prawdziwy, 1 -> Fałszywy).

data_banknotes = np.loadtxt("data_banknote_authentication.txt", delimiter=",")
y_banknotes = data_banknotes[:, -1].astype(int)
X_banknotes = data_banknotes[:, :-1]
# X_banknotes = X_banknotes[:, [0, 2]]  # Tutaj pierwsza i trzecia cecha.

feature_names_banknotes = [
	"Wariancja",
	"Skośność",
	"Kurtoza",
	"Entropia"
]
# feature_names_banknotes = [feature_names_banknotes[i] for i in [0, 2]]
class_names_banknotes = ["Prawdziwy", "Fałszywy"]

print("Zbiór danych: Banknote Authentication (Uwierzytelnianie Banknotów)")
print(f"Liczba próbek: {X_banknotes.shape[0]}")
print(f"Liczba cech: {X_banknotes.shape[1]}")
print(f"Klasy: {class_names_banknotes}\n")

# --------------------------------------------------
# Podział danych na zbiory treningowe i testowe.
# --------------------------------------------------

# Funkcja train_test_split losowo dzieli dane:
# 	- 70% próbek na trening (do nauki modelu),
# 	- 30% próbek na test (do sprawdzenia jakości).

X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds = train_test_split(
	X_seeds, y_seeds,
	test_size=0.3, stratify=y_seeds, random_state=5
)

X_train_banknotes, X_test_banknotes, y_train_banknotes, y_test_banknotes = train_test_split(
	X_banknotes, y_banknotes,
	test_size=0.3, stratify=y_banknotes, random_state=5
)

# --------------------------------------------------
# Skalowanie danych.
# --------------------------------------------------

scaler_seeds = StandardScaler()  # Tworzenie obiektu scalera.
scaler_seeds = scaler_seeds.fit(X_train_seeds)  # Uczenie (fitowanie) scalera na danych treningowych.
X_train_scaled_seeds = scaler_seeds.transform(X_train_seeds)  # Skalowanie danych treningowych przy użyciu wyuczonego scalera.
X_test_scaled_seeds = scaler_seeds.transform(X_test_seeds)  # Skalowanie danych testowych przy użyciu wyuczonego scalera.

scaler_banknotes = StandardScaler()  # Tworzenie obiektu scalera.
scaler_banknotes = scaler_banknotes.fit(X_train_banknotes)  # Uczenie (fitowanie) scalera na danych treningowych.
X_train_scaled_banknotes = scaler_banknotes.transform(X_train_banknotes)  # Skalowanie danych treningowych przy użyciu wyuczonego scalera.
X_test_scaled_banknotes = scaler_banknotes.transform(X_test_banknotes)  # Skalowanie danych testowych przy użyciu wyuczonego scalera.

# --------------------------------------------------
# Definiowanie hiperparametrów oraz list do zapisu wyników.
# --------------------------------------------------

# ks = (3, )
ks = (3, 5)  # Ile najbliższych sąsiadów jest branych pod uwagę przy klasyfikacji.
# k_inits = (3, )
k_inits = (3, 5)  # Ile sąsiadów jest używanych do startowego wyznaczenia przynależności dla próbek treningowych.
# ms = (2, )
ms = (2, 5, 10)  # Parametr rozmycia, im większy, tym bardziej miękkie przynależności.

algorithms = []
datasets = []
scalings = []
predictions = []
accuracies = []
confusion_matrices = []

# --------------------------------------------------
# Testowanie klasycznego kNN i rozmytego Fuzzy kNN.
# --------------------------------------------------

for dataset_name in ("Seeds", "Banknote Authentication"):
	for k in ks:
		for scaled_value in (True, False):

			# --------------------------------------------------
			# Uczenie klasycznego kNN.
			# --------------------------------------------------

			# Klasyczny kNN:
			# Tworzenie obiektu klasyfikatora z 'n' sąsiadami. Oznacza to,
			# że dla każdej nowej próbki algorytm patrzy na 'n' najbliższych sąsiadów
			# w danych treningowych i wybiera klasę większościową.

			knn = KNeighborsClassifier(n_neighbors=k)
			knn_predicted, knn_accuracy, knn_cm = test_classifier(knn, dataset=dataset_name, scaled=scaled_value)

			algorithms.append(f'klasyczny kNN dla k={k}, scaled={scaled_value}')
			datasets.append(dataset_name)
			scalings.append(scaled_value)
			predictions.append(knn_predicted)
			accuracies.append(knn_accuracy)
			confusion_matrices.append(knn_cm)

			# --------------------------------------------------
			# Kroswalidacja.
			# --------------------------------------------------

			X, y, roc_auc = (X_seeds, y_seeds, 'roc_auc_ovo') if dataset_name == "Seeds" else (X_banknotes, y_banknotes, 'roc_auc')
			skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

			if scaled_value:
				estimator_knn = Pipeline([
					('scaler', StandardScaler()),
					('knn', knn)
				])
			else:
				estimator_knn = knn

			knn_cv_accuracy = cross_val_score(cv=skf, estimator=estimator_knn, X=X, y=y, scoring='accuracy')
			knn_cv_roc_auc = cross_val_score(cv=skf, estimator=estimator_knn, X=X, y=y, scoring=roc_auc)

			# --------------------------------------------------
			# Ocena jakości i skuteczności klasyfikatora.
			# --------------------------------------------------

			x_train_data, x_test_data, y_train_data, y_test_data = get_data(dataset_name, scaled_value)  # Pobranie danych.
			print(f'\n\nZbiór danych: {dataset_name}\n')
			print(f'\tEtykiety rzeczywiste (pierwsze 25 danych z testu): {one_line(y_test_data[:25])}\n')

			print(f'\t\tEtykiety przypisane przez klasyczny kNN dla k={k}, scaled={scaled_value} (pierwsze 25 danych z testu): {one_line(knn_predicted[:25])}')
			print_result(knn_accuracy, knn_cm, knn_cv_accuracy, knn_cv_roc_auc)

			for k_init in k_inits:
				for m in ms:

					# --------------------------------------------------
					# Uczenie rozmytego Fuzzy kNN.
					# --------------------------------------------------

					# Rozmyty Fuzzy kNN:
					# Tworzenie obiektu klasyfikatora z 'n' sąsiadami. Oznacza to,
					# że dla każdej nowej próbki algorytm patrzy na 'n' najbliższych sąsiadów
					# w danych treningowych i określa stopień przynależności do każdej z klas.

					fuzzy_knn = FuzzyKNN(k=k, k_init=k_init, m=m)
					fuzzy_knn_predicted, fuzzy_knn_accuracy, fuzzy_knn_cm = test_classifier(fuzzy_knn, dataset=dataset_name, scaled=scaled_value)

					algorithms.append(f'rozmyty Fuzzy kNN dla k={k}, scaled={scaled_value}, k_init={k_init}, m={m}')
					datasets.append(dataset_name)
					scalings.append(scaled_value)
					predictions.append(fuzzy_knn_predicted)
					accuracies.append(fuzzy_knn_accuracy)
					confusion_matrices.append(fuzzy_knn_cm)

					# --------------------------------------------------
					# Kroswalidacja.
					# --------------------------------------------------

					if scaled_value:
						estimator_fuzzy_knn = Pipeline([
							('scaler', StandardScaler()),
							('fuzzy', fuzzy_knn)
						])
					else:
						estimator_fuzzy_knn = fuzzy_knn

					fuzzy_knn_cv_accuracy = cross_val_score(cv=skf, estimator=estimator_fuzzy_knn, X=X, y=y, scoring='accuracy')
					fuzzy_knn_cv_roc_auc = cross_val_score(cv=skf, estimator=estimator_fuzzy_knn, X=X, y=y, scoring=roc_auc)

					# --------------------------------------------------
					# Ocena jakości i skuteczności klasyfikatora.
					# --------------------------------------------------

					print(f'\t\tEtykiety przypisane przez rozmyty Fuzzy kNN dla k={k}, scaled={scaled_value}, k_init={k_init}, m={m} (pierwsze 25 danych z testu): {one_line(fuzzy_knn_predicted[:25])}')
					print_result(fuzzy_knn_accuracy, fuzzy_knn_cm, fuzzy_knn_cv_accuracy, fuzzy_knn_cv_roc_auc)

# --------------------------------------------------
# Analiza najlepszych i najgorszych konfiguracji.
# --------------------------------------------------

results = list(zip(
	algorithms,  # Opis konfiguracji algorytmu.
	datasets,  # Nazwa zbioru.
	scalings,  # Skalowanie danych (True/False).
	predictions,  # Przewidziane etykiety dla testu.
	accuracies,  # Dokładność dla testu.
	confusion_matrices  # Macierz pomyłek dla testu.
))

for dataset_name in ("Seeds", "Banknote Authentication"):
	# Filtrowanie wyników dla danego zbioru danych.
	dataset_results = [r for r in results if r[1] == dataset_name]
	print(f"\n\nZbiór danych: {dataset_name}\n")

	# Podział na klasyczny kNN i rozmyty Fuzzy kNN.
	knn_results = [r for r in dataset_results if r[0].startswith('klasyczny kNN')]
	fuzzy_knn_results = [r for r in dataset_results if r[0].startswith('rozmyty Fuzzy kNN')]

	# Sortowanie po accuracy (dokładność klasyfikatora).
	knn_sorted = sorted(knn_results, key=lambda x: x[4])  # Sortowanie rosnąco (od najgorszego do najlepszego).
	fuzzy_knn_sorted = sorted(fuzzy_knn_results, key=lambda x: x[4])  # Sortowanie rosnąco (od najgorszego do najlepszego).

	# Najlepsze i najgorsze konfiguracje według accuracy (dokładności klasyfikatora).
	knn_best = knn_sorted[-1]
	knn_worst = knn_sorted[0]
	fuzzy_knn_best = fuzzy_knn_sorted[-1]
	fuzzy_knn_worst = fuzzy_knn_sorted[0]

	# --------------------------------------------------
	# Wyniki dla klasycznego kNN.
	# --------------------------------------------------

	print('\tKlasyczny kNN – Najlepsza konfiguracja:')
	print_best_or_worst_classifier(knn_best[0], knn_best[4], knn_best[5])

	print('\tKlasyczny kNN – Najgorsza konfiguracja:')
	print_best_or_worst_classifier(knn_worst[0], knn_worst[4], knn_worst[5])

	# --------------------------------------------------
	# Wyniki dla rozmytego Fuzzy kNN.
	# --------------------------------------------------

	print('\tRozmyty Fuzzy kNN – Najlepsza konfiguracja:')
	print_best_or_worst_classifier(fuzzy_knn_best[0], fuzzy_knn_best[4], fuzzy_knn_best[5])

	print('\tRozmyty Fuzzy kNN – Najgorsza konfiguracja:')
	print_best_or_worst_classifier(fuzzy_knn_worst[0], fuzzy_knn_worst[4], fuzzy_knn_worst[5])

# --------------------------------------------------
# Macierz pomyłek dla wybranych danych.
# --------------------------------------------------

# Macierz pomyłek pokazuje, ile próbek z każdej klasy zostało sklasyfikowanych poprawnie
# (na przekątnej od górnej-lewej do dolnej-prawej) i błędnie (poza nią).

# Pobranie odpowiednich rekordów z bazy danych Seeds (dla k=3, scaled=True, k_init=3, m=2).
knn_k3_seeds = next(r for r in results if r[0] == 'klasyczny kNN dla k=3, scaled=True' and r[1] == 'Seeds')
fuzzy_knn_k3_k_init3_m2_seeds = next(r for r in results if r[0] == 'rozmyty Fuzzy kNN dla k=3, scaled=True, k_init=3, m=2' and r[1] == 'Seeds')

# Pobranie odpowiednich rekordów z bazy danych Banknote Authentication (dla k=3, scaled=True, k_init=3, m=2).
knn_k3_banknotes = next(r for r in results if r[0] == 'klasyczny kNN dla k=3, scaled=True' and r[1] == 'Banknote Authentication')
fuzzy_knn_k3_k_init3_m2_banknotes = next(r for r in results if r[0] == 'rozmyty Fuzzy kNN dla k=3, scaled=True, k_init=3, m=2' and r[1] == 'Banknote Authentication')

# Wydobycie macierzy pomyłek z bazy danych Seeds (dla k=3, scaled=True, k_init=3, m=2).
knn_cm_seeds = knn_k3_seeds[5]
fuzzy_knn_cm_seeds = fuzzy_knn_k3_k_init3_m2_seeds[5]

# Wydobycie macierzy pomyłek z bazy danych Banknote Authentication (dla k=3, scaled=True, k_init=3, m=2).
knn_cm_banknotes = knn_k3_banknotes[5]
fuzzy_knn_cm_banknotes = fuzzy_knn_k3_k_init3_m2_banknotes[5]

# Przygotowanie wizualizacji macierzy pomyłek.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
font_size = 10

# Seeds — Klasyczny kNN.
disp_seeds_knn = ConfusionMatrixDisplay(
	confusion_matrix=knn_cm_seeds,
	display_labels=class_names_seeds
)
disp_seeds_knn.plot(ax=axes[0, 0], values_format="d", colorbar=False)
axes[0, 0].set_title("Seeds — Klasyczny kNN", fontsize=font_size)
axes[0, 0].set_xlabel("Etykiety przewidziane", fontsize=font_size)
axes[0, 0].set_ylabel("Etykiety rzeczywiste", fontsize=font_size)

# Seeds — Rozmyty Fuzzy kNN.
disp_seeds_fuzzy = ConfusionMatrixDisplay(
	confusion_matrix=fuzzy_knn_cm_seeds,
	display_labels=class_names_seeds
)
disp_seeds_fuzzy.plot(ax=axes[0, 1], values_format="d", colorbar=False)
axes[0, 1].set_title("Seeds — Rozmyty Fuzzy kNN", fontsize=font_size)
axes[0, 1].set_xlabel("Etykiety przewidziane", fontsize=font_size)
axes[0, 1].set_ylabel("Etykiety rzeczywiste", fontsize=font_size)

# Banknote Authentication — Klasyczny kNN.
disp_bank_knn = ConfusionMatrixDisplay(
	confusion_matrix=knn_cm_banknotes,
	display_labels=class_names_banknotes
)
disp_bank_knn.plot(ax=axes[1, 0], values_format="d", colorbar=False)
axes[1, 0].set_title("Banknote Authentication — Klasyczny kNN", fontsize=font_size)
axes[1, 0].set_xlabel("Etykiety przewidziane", fontsize=font_size)
axes[1, 0].set_ylabel("Etykiety rzeczywiste", fontsize=font_size)

# Banknote Authentication – Rozmyty Fuzzy kNN.
disp_bank_fuzzy = ConfusionMatrixDisplay(
	confusion_matrix=fuzzy_knn_cm_banknotes,
	display_labels=class_names_banknotes
)
disp_bank_fuzzy.plot(ax=axes[1, 1], values_format="d", colorbar=False)
axes[1, 1].set_title("Banknote Authentication – Rozmyty Fuzzy kNN", fontsize=font_size)
axes[1, 1].set_xlabel("Etykiety przewidziane", fontsize=font_size)
axes[1, 1].set_ylabel("Etykiety rzeczywiste", fontsize=font_size)

# Wizualizacja macierzy pomyłek.
fig.suptitle(
	"Porównanie macierzy pomyłek klasycznego kNN dla k=3, scaled=True i rozmytego Fuzzy kNN dla k=3, scaled=True, k_init=3, m=2",
	fontsize=font_size
)
plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.show()
print()

# --------------------------------------------------
# Wizualizacja danych na wykresie.
# --------------------------------------------------

# Wydobycie etykiet przewidzianych z bazy danych Seeds (dla k=3, scaled=True, k_init=3, m=2).
knn_predicted_seeds = knn_k3_seeds[3]
fuzzy_knn_predicted_seeds = fuzzy_knn_k3_k_init3_m2_seeds[3]

# Wydobycie etykiet przewidzianych z bazy danych Banknote Authentication (dla k=3, scaled=True, k_init=3, m=2).
knn_predicted_banknotes = knn_k3_banknotes[3]
fuzzy_knn_predicted_banknotes = fuzzy_knn_k3_k_init3_m2_banknotes[3]

# Mapy kolorów.
cmap_seeds = ListedColormap([
	[0.267004, 0.004874, 0.329415, 1.0],
	[0.128729, 0.563265, 0.551229, 1.0],
	[0.993248, 0.906157, 0.143936, 1.0]
])  # (0 -> Kama, 0.5 -> Rosa, 1 -> Canadian).
cmap_banknotes = ListedColormap([
	[0.267004, 0.004874, 0.329415, 1.0],
	[0.993248, 0.906157, 0.143936, 1.0]
])  # (0 -> Prawdziwy, 1 -> Fałszywy).

# Przygotowanie wizualizacji porównania klas rzeczywistych, klas klasycznego kNN, klas rozmytego Fuzzy kNN.
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Seeds — Etykiety rzeczywiste.
axes[0, 0].scatter(
	X_test_seeds[:, 0], X_test_seeds[:, 1],
	c=y_test_seeds, cmap=cmap_seeds
)
axes[0, 0].set_title("Seeds — Etykiety rzeczywiste", fontsize=font_size)
axes[0, 0].set_xlabel(feature_names_seeds[0], fontsize=font_size)
axes[0, 0].set_ylabel(feature_names_seeds[1], fontsize=font_size)

# Seeds — Etykiety klasycznego kNN.
axes[0, 1].scatter(
	X_test_seeds[:, 0], X_test_seeds[:, 1],
	c=knn_predicted_seeds, cmap=cmap_seeds
)
axes[0, 1].set_title("Seeds — Etykiety klasycznego kNN", fontsize=font_size)
axes[0, 1].set_xlabel(feature_names_seeds[0], fontsize=font_size)
axes[0, 1].set_ylabel(feature_names_seeds[1], fontsize=font_size)

# Seeds — Etykiety rozmytego Fuzzy kNN.
axes[0, 2].scatter(
	X_test_seeds[:, 0], X_test_seeds[:, 1],
	c=fuzzy_knn_predicted_seeds, cmap=cmap_seeds
)
axes[0, 2].set_title("Seeds — Etykiety rozmytego Fuzzy kNN", fontsize=font_size)
axes[0, 2].set_xlabel(feature_names_seeds[0], fontsize=font_size)
axes[0, 2].set_ylabel(feature_names_seeds[1], fontsize=font_size)

# Dodanie legendy dla Seeds.
add_legend(axes[0, 0], class_names_seeds, cmap_seeds)
add_legend(axes[0, 1], class_names_seeds, cmap_seeds)
add_legend(axes[0, 2], class_names_seeds, cmap_seeds)

# Banknote Authentication — Etykiety rzeczywiste.
axes[1, 0].scatter(
	X_test_banknotes[:, 0], X_test_banknotes[:, 1],
	c=y_test_banknotes, cmap=cmap_banknotes
)
axes[1, 0].set_title("Banknote Authentication — Etykiety rzeczywiste", fontsize=font_size)
axes[1, 0].set_xlabel(feature_names_banknotes[0], fontsize=font_size)
axes[1, 0].set_ylabel(feature_names_banknotes[1], fontsize=font_size)

# Banknote Authentication — Etykiety klasycznego kNN.
axes[1, 1].scatter(
	X_test_banknotes[:, 0], X_test_banknotes[:, 1],
	c=knn_predicted_banknotes, cmap=cmap_banknotes
)
axes[1, 1].set_title("Banknote Authentication — Etykiety klasycznego kNN", fontsize=font_size)
axes[1, 1].set_xlabel(feature_names_banknotes[0], fontsize=font_size)
axes[1, 1].set_ylabel(feature_names_banknotes[1], fontsize=font_size)

# Banknote Authentication — Etykiety rozmytego Fuzzy kNN.
axes[1, 2].scatter(
	X_test_banknotes[:, 0], X_test_banknotes[:, 1],
	c=fuzzy_knn_predicted_banknotes, cmap=cmap_banknotes
)
axes[1, 2].set_title("Banknote Authentication — Etykiety rozmytego Fuzzy kNN", fontsize=font_size)
axes[1, 2].set_xlabel(feature_names_banknotes[0], fontsize=font_size)
axes[1, 2].set_ylabel(feature_names_banknotes[1], fontsize=font_size)

# Dodanie legendy dla Banknote Authentication.
add_legend(axes[1, 0], class_names_banknotes, cmap_banknotes)
add_legend(axes[1, 1], class_names_banknotes, cmap_banknotes)
add_legend(axes[1, 2], class_names_banknotes, cmap_banknotes)

# Wizualizacja porównania klas.
fig.suptitle(
	"Porównanie etykiet rzeczywistych oraz klasycznego kNN dla k=3, scaled=True i rozmytego Fuzzy kNN dla k=3, scaled=True, k_init=3, m=2 (dla dwóch pierwszych cech)",
	fontsize=font_size
)
plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.show()

# --------------------------------------------------
# Wizualizacja rozmytych przynależności w rozmytym Fuzzy kNN.
# --------------------------------------------------

# Odtworzenie rozmytego Fuzzy kNN dla zbioru danych Seeds (dla k=3, scaled=True, k_init=3, m=2).
fuzzy_knn_seeds = FuzzyKNN(k=3, k_init=3, m=2)
fuzzy_knn_seeds.fit(X_train_scaled_seeds, y_train_seeds)
fuzzy_proba_seeds = fuzzy_knn_seeds.predict_proba(X_test_scaled_seeds)

# Przygotowanie wizualizacji przynależność do klas w rozmytym Fuzzy kNN dla zbioru danych Seeds.
fig, axes = plt.subplots(1, len(class_names_seeds), figsize=(15, 4))

for i, ax in enumerate(axes):
	sc = ax.scatter(
		X_test_seeds[:, 0],
		X_test_seeds[:, 1],
		c=fuzzy_proba_seeds[:, i],
		cmap="viridis",
		vmin=0,
		vmax=1
	)
	ax.set_title(f"Seeds — Przynależność do klasy: {class_names_seeds[i]}", fontsize=font_size)
	ax.set_xlabel(feature_names_seeds[0], fontsize=font_size)
	ax.set_ylabel(feature_names_seeds[1], fontsize=font_size)
	fig.colorbar(sc, ax=ax, label="Stopień przynależności")

# Wizualizacja przynależności do klas w rozmytym Fuzzy kNN zbioru danych Seeds.
plt.suptitle("Rozmyte przynależności próbek testowych w Fuzzy kNN zbioru Seeds (dla dwóch pierwszych cech)", fontsize=font_size)
plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.show()

# Odtworzenie rozmytego Fuzzy kNN dla zbioru danych Banknote Authentication (dla k=3, scaled=True, k_init=3, m=2).
fuzzy_knn_banknotes = FuzzyKNN(k=3, k_init=3, m=2)
fuzzy_knn_banknotes.fit(X_train_scaled_banknotes, y_train_banknotes)
fuzzy_proba_banknotes = fuzzy_knn_banknotes.predict_proba(X_test_scaled_banknotes)

# Przygotowanie wizualizacji przynależność do klas w rozmytym Fuzzy kNN dla zbioru danych Banknote Authentication.
fig, axes = plt.subplots(1, len(class_names_banknotes), figsize=(10, 4))

for i, ax in enumerate(axes):
	sc = ax.scatter(
		X_test_banknotes[:, 0],
		X_test_banknotes[:, 1],
		c=fuzzy_proba_banknotes[:, i],
		cmap="viridis",
		vmin=0,
		vmax=1
	)
	ax.set_title(f"Banknote Authentication — Przynależność do klasy: {class_names_banknotes[i]}", fontsize=font_size)
	ax.set_xlabel(feature_names_banknotes[0], fontsize=font_size)
	ax.set_ylabel(feature_names_banknotes[1], fontsize=font_size)
	fig.colorbar(sc, ax=ax, label="Stopień przynależności")

# Wizualizacja przynależności do klas w rozmytym Fuzzy kNN zbioru danych Banknote Authentication.
plt.suptitle("Rozmyte przynależności próbek testowych w Fuzzy kNN zbioru Banknote Authentication (dla dwóch pierwszych cech)", fontsize=font_size)
plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.show()

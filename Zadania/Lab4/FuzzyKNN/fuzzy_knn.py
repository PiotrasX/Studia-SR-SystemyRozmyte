import operator
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class FuzzyKNN(BaseEstimator, ClassifierMixin):
	# --------------------------------------------------
	# Inicjalizacja klasyfikatora rozmytego kNN.
	#
	# Informacje o funkcji:
	# - Parametr k określa liczbę najbliższych sąsiadów branych pod uwagę
	# 	do ustalania klasy decyzyjnej obiektów testowych.
	# -	W self.k_init następuje inicjalizacja algorytmu,
	# 	w celu rozmycia etykiet klas decyzyjnych obiektów treningowych.
	# - Parametr m ustala wpływ danego sąsiada na przynależenie do klasy decyzyjnej.
	# 	Dla niektórych wartości parametru dalsi sąsiedzi mają mniejszy wpływ na decyzję niż bliżsi.
	# --------------------------------------------------
	def __init__(self, k=3, k_init=3, m=2, plot=False):
		self.k = k
		self.k_init = k_init
		if m <= 1:
			raise ValueError('Parametr m musi być większy niż 1')
		self.m = m
		self.plot = plot
		self.fitted_ = False
		self._estimator_type = "classifier"

	# --------------------------------------------------
	# Trenowanie modelu na danych.
	#
	# Informacje o funkcji:
	# - Parametr self.classes to lista unikalnych klas.
	# - W self.df tworzy się obiekt DataFrame z kolumną 'y' (etykietami).
	# - W self.memberships wykonywane jest rozmycie etykiet klas decyzyjnych obiektów treningowych.
	# --------------------------------------------------
	def fit(self, X, y=None):
		self._check_params(X, y)
		self.X = X
		self.y = y

		self.xdim = len(self.X[0])
		self.n = len(y)

		classes = list(set(y))
		classes.sort()
		self.classes = classes

		self.df = pd.DataFrame(self.X)
		self.df['y'] = self.y

		self.memberships = self._compute_memberships()
		self.df['membership'] = self.memberships

		self.fitted_ = True

		return self

	# --------------------------------------------------
	# Właściwe przewidywanie z informacją o głosach sąsiadów.
	# --------------------------------------------------
	def predict_original(self, X):
		if not self.fitted_:
			raise Exception('Najpierw wywołaj fit() przed predict()')

		y_predicted = []

		for x in X:
			neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x, self.k)
			votes = {}

			for c in self.classes:
				den = 0
				for n in range(self.k):
					dist = np.linalg.norm(x - neighbors.iloc[n, 0:self.xdim])
					den += 1 / (dist ** (2 / (self.m - 1)) if dist != 0 else np.finfo(np.float64).eps)

				neighbors_votes = []
				for n in range(self.k):
					dist = np.linalg.norm(x - neighbors.iloc[n, 0:self.xdim])
					num = neighbors.iloc[n].membership[c] / (dist ** (2 / (self.m - 1)) if dist != 0 else np.finfo(np.float64).eps)
					neighbors_votes.append(num / den)

				votes[c] = np.sum(neighbors_votes)

			predicted = max(votes.items(), key=operator.itemgetter(1))[0]
			y_predicted.append((predicted, votes))

		return y_predicted

	# --------------------------------------------------
	# Zwracanie tylko etykiety (klasy 0/1).
	# --------------------------------------------------
	def predict(self, X):
		return np.array([p[0] for p in self.predict_original(X)])

	# --------------------------------------------------
	# Zwracanie stopni przynależności (fuzzy-prawdopodobieństwa).
	# --------------------------------------------------
	def predict_proba(self, X):
		output = self.predict_original(X)
		dec = []
		for o in output:
			total = sum(o[1].values())
			instance = [o[1][c] / total if total != 0 else 0 for c in self.classes]
			dec.append(instance)

		return np.array(dec)

	# --------------------------------------------------
	# Obliczanie dokładności modelu.
	# --------------------------------------------------
	def score(self, X, y, **kwargs):
		if not self.fitted_:
			raise Exception('Najpierw wywołaj fit() przed score()')

		return accuracy_score(y_true=y, y_pred=self.predict(X))

	# --------------------------------------------------
	# Znajdowanie najbliższych sąsiadów.
	# --------------------------------------------------
	def _find_k_nearest_neighbors(self, df, x, k):
		df['distances'] = [np.linalg.norm(df.iloc[i, 0:self.xdim] - x) for i in range(self.n)]
		df.sort_values(by='distances', ascending=True, inplace=True)
		return df.iloc[0:k]

	# --------------------------------------------------
	# Liczenie liczby sąsiadów w każdej klasie.
	# --------------------------------------------------
	def _get_counts(self, neighbors):
		groups = neighbors.groupby('y')
		return {group[0]: len(group[1]) for group in groups}

	# --------------------------------------------------
	# Wyznaczanie początkowych przynależności rozmytych.
	# --------------------------------------------------
	def _compute_memberships(self):
		memberships = []
		for i in range(self.n):
			x = self.X[i]
			y = self.y[i]

			neighbors = self._find_k_nearest_neighbors(pd.DataFrame.copy(self.df), x, self.k_init)
			counts = self._get_counts(neighbors)

			membership = dict()
			for c in self.classes:
				try:
					uci = 0.49 * (counts[c] / self.k_init)
					if c == y:
						uci += 0.51
					membership[c] = uci
				except:
					membership[c] = 0

			memberships.append(membership)

		return memberships

	# --------------------------------------------------
	# Walidowanie parametrów.
	# --------------------------------------------------
	def _check_params(self, X, y):
		if not isinstance(self.k, int):
			raise Exception('"k" musi być typu int')
		if self.k >= len(y):
			raise Exception('"k" musi być mniejsze niż liczba próbek treningowych')
		if not isinstance(self.plot, bool):
			raise Exception('"plot" musi być typu bool')

	# --------------------------------------------------
	# Zwracanie listy unikalnych klas modelu.
	# --------------------------------------------------
	@property
	def classes_(self):
		return np.array(self.classes)

	# --------------------------------------------------
	# Definiowanie podstawowych znaczników scikit-learn.
	# --------------------------------------------------
	def _more_tags(self):
		return {"binary_only": True, "requires_y": True}

	# --------------------------------------------------
	# Informowanie scikit-learn, czy model został wytrenowany.
	# --------------------------------------------------
	def __sklearn_is_fitted__(self):
		return hasattr(self, "fitted_") and self.fitted_ is True

	# --------------------------------------------------
	# Ustawianie metadanych modelu scikit-learn.
	# --------------------------------------------------
	def __sklearn_tags__(self):
		return SimpleNamespace(
			estimator_type="classifier",
			requires_fit=True,
			requires_y=True,
			binary_only=False,
			allow_nan=False,
		)

pass

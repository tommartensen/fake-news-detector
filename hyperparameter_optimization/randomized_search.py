# build a classifier
import json
import random
import os
from time import time

import numpy as np
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")


def report(results, n_top=3):
	# Utility function to report best scores
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank:" + str(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				  results['mean_test_score'][candidate],
				  results['std_test_score'][candidate]))
			print("Parameters: {0}\n".format(results['params'][candidate]))


def get_random_layer_sizes():
	hidden_layers = []
	depth = random.randint(1, 6)
	for i in range(0, depth):
		hidden_layers.append(random.randint(20, 300))
	return tuple(hidden_layers)


def run_search(clf, param_dist, X, y):
	# run randomized search, due to memory limits on developer machine, RandomizedSearchCV must run single-threaded.
	print("Starting randomized search for", type(clf).__name__, "...")
	n_iter_search = 50
	random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=1,
	                                   cv=5, scoring=make_scorer(f1_score, labels=[0, 1]))

	start = time()
	random_search.fit(X, y)
	print("Random search took %.2f seconds, %d parameter settings tested." % ((time() - start), n_iter_search))
	report(random_search.cv_results_)


def main(filename):
	print("Loading data...")
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/" + filename), "r") as f:
		X = json.load(f)
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/labels.json"), "r") as f:
		y = json.load(f)

	n_features = len(X[0])
	configurations = [
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"max_depth": sp_randint(1, n_features),
				"max_features": sp_randint(100, n_features),
				"min_samples_split": sp_randint(200, n_features),
				"min_samples_leaf": sp_randint(100, n_features),
				"bootstrap": [True, False],
				"criterion": ["gini", "entropy"]
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
				"penalty": ["l2"],
				"dual": [0],
				"C": np.random.uniform(low=0.01, high=5, size=(200,)),
				"class_weight": ["balanced", None]
			}
		],
		[
			MLPClassifier(),
			{
				"activation": ["identity", "logistic", "tanh", "relu"],
				"solver": ["lbfgs", "sgd", "adam"],
				"alpha": np.random.uniform(low=0.000001, high=0.01, size=(200,)),
				"tol": np.random.uniform(low=0.000001, high=0.01, size=(200,)),
				"hidden_layer_sizes": get_random_layer_sizes()
			}
		]
	]

	for clf, param_dist in configurations:
		run_search(clf, param_dist, X, y)


if __name__ == "__main__":
	for i in range(1, 4):
		print("ngram: " + str(i))
		main("ngram_l%d_u%d.json" % (i, i))
		print("ngram: " + str(i) + " tfidf")
		main("ngram_l%d_u%d_t.json" % (i, i))
		print("hashed: " + str(i))
		main("hashed_10000_l" + str(i) + "_u" + str(i) + ".json")
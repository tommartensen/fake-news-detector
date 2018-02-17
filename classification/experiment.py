import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def load_training_data(vectorizer_info):
	number = vectorizer_info["number"]
	hashed = vectorizer_info["hashed"]
	tfidf = vectorizer_info["tfidf"]
	if hashed:
		filename = "hashed_10000_l" + str(number) + "_u" + str(number)
	else:
		filename = "ngram_l" + str(number) + "_u" + str(number)
		if tfidf:
			filename += "_t"
	print("The current vectorizer configuration is: " + filename)

	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/trained/" + filename + ".json"), "r") as f:
		training_data = json.load(f)
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/data/trained/labels_training.json"), "r") as f:
		training_labels = json.load(f)
	print("Training data loading completed...")
	return training_data, training_labels


def load_test_data(vectorizer_info):
	number = vectorizer_info["number"]
	hashed = vectorizer_info["hashed"]
	tfidf = vectorizer_info["tfidf"]
	if hashed:
		filename = "hashed_10000_test_l" + str(number) + "_u" + str(number) + ".json"
	else:
		filename = "ngram_test_l" + str(number) + "_u" + str(number)
		if tfidf:
			filename += "_t"
		filename += ".json"
	with open(os.path.join(os.path.dirname(__file__), "../feature_regeneration/data/" + filename), "r") as f:
		test_data = json.load(f)
	with open(os.path.join(os.path.dirname(__file__), "../feature_regeneration/data/labels_test.json"), "r") as f:
		test_labels = json.load(f)
	print("Test data loading completed...")
	return test_data, test_labels


def print_score(actual_labels, predicted_labels):
	print("F1 score: " + str(f1_score(y_true=actual_labels, y_pred=predicted_labels, labels=[0, 1])))
	tn, fp, fn, tp = confusion_matrix(y_true=actual_labels, y_pred=predicted_labels).ravel()
	print("TN:", tn, "\tFP:", fp, "\tFN:", fn, "\tTP:", tp)
	print("Precision:", (tp / (tp + fp)))
	print("Recall:", (tp / (tp + fn)))


if __name__ == "__main__":
	configurations = [
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "liblinear",
				"penalty": "l2",
				"dual": 0,
				"C": 0.19675145,
				"class_weight": "balanced"
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "liblinear",
				"penalty": "l2",
				"dual": 0,
				"C": 3.20842543,
				"class_weight": "balanced"
			},
			{
				"number": 1,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "lbfgs",
				"penalty": "l2",
				"dual": 0,
				"C": 3.9063106,
				"class_weight": None
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "saga",
				"penalty": "l2",
				"dual": 0,
				"C": 1.1300675,
				"class_weight": "balanced"
			},
			{
				"number": 2,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "saga",
				"penalty": "l2",
				"dual": 0,
				"C": 2.45229817,
				"class_weight": "balanced"
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "saga",
				"penalty": "l2",
				"dual": 0,
				"C": 0.5893172,
				"class_weight": "balanced"
			},
			{
				"number": 3,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "relu",
				"solver": "adam",
				"alpha": 0.0065998,
				"tol": 0.00652403,
				"hidden_layer_sizes": 298
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "relu",
				"solver": "adam",
				"alpha": 0.00465253581,
				"tol": 0.0018537,
				"hidden_layer_sizes": 128
			},
			{
				"number": 1,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "logistic",
				"solver": "adam",
				"alpha": 0.00154166,
				"tol": 0.0056047,
				"hidden_layer_sizes": 241
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "tanh",
				"solver": "lbfgs",
				"alpha": 0.00059665,
				"tol": 0.001043,
				"hidden_layer_sizes": 156
			},
			{
				"number": 2,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "relu",
				"solver": "adam",
				"alpha": 0.0025847,
				"tol": 0.0033457,
				"hidden_layer_sizes": 161
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "identity",
				"solver": "adam",
				"alpha": 0.0074729,
				"tol": 0.0036438,
				"hidden_layer_sizes": 33
			},
			{
				"number": 3,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "gini",
				"max_depth": 1120,
				"max_features": 3595,
				"min_samples_leaf": 46,
				"min_samples_split": 7262
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "entropy",
				"max_depth": 646,
				"max_features": 3096,
				"min_samples_leaf": 103,
				"min_samples_split": 2279
			},
			{
				"number": 1,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "gini",
				"max_depth": 540,
				"max_features": 1067,
				"min_samples_leaf": 173,
				"min_samples_split": 3204
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "entropy",
				"max_depth": 4337,
				"max_features": 1301,
				"min_samples_leaf": 278,
				"min_samples_split": 3381
			},
			{
				"number": 2,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "gini",
				"max_depth": 959,
				"max_features": 1451,
				"min_samples_leaf": 110,
				"min_samples_split": 1463
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "entropy",
				"max_depth": 1812,
				"max_features": 685,
				"min_samples_leaf": 134,
				"min_samples_split": 431
			},
			{
				"number": 3,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 1,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 2,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 0,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 3,
				"tfidf": 1,
				"hashed": 0,
			}
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 1,
				"criterion": "gini",
				"max_depth": 5312,
				"max_features": 6142,
				"min_samples_leaf": 249,
				"min_samples_split": 5151
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 1,
			},
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "entropy",
				"max_depth": 3279,
				"max_features": 7683,
				"min_samples_leaf": 263,
				"min_samples_split": 4406
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 1,
			},
		],
		[
			RandomForestClassifier(n_estimators=20, n_jobs=-1),
			{
				"bootstrap": 0,
				"criterion": "entropy",
				"max_depth": 8754,
				"max_features": 6386,
				"min_samples_leaf": 8356,
				"min_samples_split": 8313
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 1,
			},
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "liblinear",
				"penalty": "l2",
				"dual": 0,
				"C": 3.186873,
				"class_weight": "balanced"
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "lbfgs",
				"penalty": "l2",
				"dual": 0,
				"C": 0.51307969,
				"class_weight": "balanced"
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			LogisticRegression(n_jobs=-1, max_iter=100, multi_class="ovr"),
			{
				"solver": "newton-cg",
				"penalty": "l2",
				"dual": 0,
				"C": 0.01850533,
				"class_weight": "balanced"
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			GaussianNB(),
			{
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "relu",
				"solver": "adam",
				"alpha": 0.0065998,
				"tol": 0.000156,
				"hidden_layer_sizes": 167
			},
			{
				"number": 1,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "identity",
				"solver": "adam",
				"alpha": 0.0074729,
				"tol": 0.006715,
				"hidden_layer_sizes": 110
			},
			{
				"number": 2,
				"tfidf": 0,
				"hashed": 1,
			}
		],
		[
			MLPClassifier(),
			{
				"activation": "relu",
				"solver": "lbfgs",
				"alpha": 0.005586,
				"tol": 0.001587,
				"hidden_layer_sizes": 173
			},
			{
				"number": 3,
				"tfidf": 0,
				"hashed": 1,
			}
		]
	]
	for clf, params, vectorizer_info in configurations:
		print("Evaluating performance for " + type(clf).__name__ + "...")
		training_data, training_labels = load_training_data(vectorizer_info)
		clf.set_params(**params)

		print("Fitting model with training data...")
		clf.fit(training_data, training_labels)

		print("Predicting labels from test data...")
		test_data, test_labels = load_test_data(vectorizer_info)
		predictions = clf.predict(test_data)

		print_score(actual_labels=test_labels, predicted_labels=predictions)
		print("-----------------------------------------------------------")
		training_data, training_labels, test_labels, test_data = None, None, None, None

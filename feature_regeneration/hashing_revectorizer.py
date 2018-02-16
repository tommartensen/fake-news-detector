import getopt
import json
import os
import pickle

import sys


def load_vectorizer(lower_bound, upper_bound):
	filename = "hashed_10000_train_l" + lower_bound + "_u" + upper_bound + ".vec"
	with open(os.path.join(os.path.dirname(__file__), "../feature_generation/vectorizers/" + filename), "rb") as f:
		return pickle.load(f)


def load_test_dataset():
	with open(os.path.join(os.path.dirname(__file__), "../preprocessing/data/test_set.json"), "r") as f:
		data = json.load(f)
	articles = []
	labels = []
	for item in data:
		articles.append(item[0])
		labels.append(item[1])
	return articles, labels


def dump_labels(labels):
	with open("data/labels_test.json", "w") as f:
		json.dump(labels, f)


def dump_features(features, lower_bound, upper_bound):
	filename = "hashed_10000_test_l" + lower_bound + "_u" + upper_bound + ".json"
	with open("data/" + filename, "w") as f:
		json.dump(features.tolist(), f)


def process_args(argv):
	if len(argv) < 2:
		print('hashing_revectorizer.py -l <lower bound> -u <upper bound>')
		sys.exit(2)
	lower_bound = 1
	upper_bound = 1
	try:
		opts, args = getopt.getopt(argv, "htl:u:", ["lower_bound=", "upper_bound="])
	except getopt.GetoptError:
		print('hashing_revectorizer.py -l <lower bound> -u <upper bound>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('hashing_revectorizer.py -l <lower bound> -u <upper bound>')
			sys.exit()
		elif opt in ("-l", "--lower_bound"):
			lower_bound = str(int(arg))
		elif opt in ("-u", "--upper_bound"):
			upper_bound = str(int(arg))
	return lower_bound, upper_bound


if __name__ == "__main__":
	lower_bound, upper_bound = process_args(sys.argv[1:])
	print("Loading vectorizer...")
	hv = load_vectorizer(lower_bound, upper_bound)
	print("Loading test dataset...")
	articles, labels = load_test_dataset()
	dump_labels(labels)
	print("Transforming features...")
	features = hv.transform(articles).toarray()
	print("Dumping features...")
	dump_features(features, lower_bound, upper_bound)

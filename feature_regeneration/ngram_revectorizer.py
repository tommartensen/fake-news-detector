import getopt
import json
import os
import pickle

import sys


def load_vectorizer(lower_bound, upper_bound, include_tfidf):
	filename = "ngram_l" + lower_bound + "_u" + upper_bound
	if include_tfidf:
		filename += "_t"
	filename += ".vec"
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
	with open(os.path.join(os.path.dirname(__file__), "../feature_regeneration/data/labels_test.json"), "w") as f:
		json.dump(labels, f)


def dump_features(features, lower_bound, upper_bound, include_tfidf):
	filename = "ngram_test_l" + lower_bound + "_u" + upper_bound
	if include_tfidf:
		filename += "_t"
	filename += ".json"
	with open(os.path.join(os.path.dirname(__file__), "../feature_regeneration/data/" + filename + ".json"), "w") as f:
		json.dump(features.tolist(), f)


def process_args(argv):
	if len(argv) < 2:
		print('ngram_revectorizer.py -l <lower bound> -u <upper bound> [-t]')
		sys.exit(2)
	lower_bound = 1
	upper_bound = 1
	include_tfidf = 0
	try:
		opts, args = getopt.getopt(argv, "htl:u:", ["lower_bound=", "upper_bound="])
	except getopt.GetoptError:
		print('ngram_revectorizer.py -l <lower bound> -u <upper bound> [-t]')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('ngram_revectorizer.py -l <lower bound> -u <upper bound> [-t]')
			sys.exit()
		elif opt in ("-l", "--lower_bound"):
			lower_bound = str(int(arg))
		elif opt in ("-u", "--upper_bound"):
			upper_bound = str(int(arg))
		elif opt in "-t":
			include_tfidf = 1
	return lower_bound, upper_bound, include_tfidf


if __name__ == "__main__":
	lower_bound, upper_bound, include_tfidf = process_args(sys.argv[1:])
	print("Loading vectorizer...")
	vec = load_vectorizer(lower_bound, upper_bound, include_tfidf)
	print("Loading test dataset...")
	articles, labels = load_test_dataset()
	dump_labels(labels)
	print("Transforming features...")
	features = vec.transform(articles).toarray()
	print("Dumping features...")
	dump_features(features, lower_bound, upper_bound, include_tfidf)

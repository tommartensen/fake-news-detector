import getopt
import json
import os
import pickle

import sys

import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def calc_df(upper_bound, size):
	"""
	Helper function to constrain the length of the vector.
	:param upper_bound: ngram upper bound
	:param size: size of the data set
	:return:
	"""
	return math.floor((size / upper_bound) * (30 / 13000))


def dump_vectorizer(vectorizer, filename):
	print("Dumping vectorizer...")
	if not os.path.exists("vectorizers"):
		os.mkdir("vectorizers")
	with open("vectorizers/" + filename + ".vec", "wb") as f:
		pickle.dump(vectorizer, f)


def main(argv):
	if len(argv) < 2:
		print('ngram_vectorizer.py -l <lower bound> -u <upper bound> [-t]')
		sys.exit(2)
	lower_bound = 1
	upper_bound = 1
	include_tfidf = 0
	try:
		opts, args = getopt.getopt(argv, "htl:u:", ["lower_bound=", "upper_bound="])
	except getopt.GetoptError:
		print('ngram_vectorizer.py -l <lower bound> -u <upper bound> [-t]')
		sys.exit(2)
	filename = "ngram"
	for opt, arg in opts:
		if opt == '-h':
			print('ngram_vectorizer.py -l <lower bound> -u <upper bound> [-t]')
			sys.exit()
		elif opt in ("-l", "--lower_bound"):
			filename += "_l" + arg
			lower_bound = int(arg)
		elif opt in ("-u", "--upper_bound"):
			filename += "_u" + arg
			upper_bound = int(arg)
		elif opt in "-t":
			filename += "_t"
			include_tfidf = 1

	articles = []
	labels = []

	print("Preparing data...")
	with open("raw_data/validation_set.json", "r") as f:
		data = json.load(f)
		for article in data:
			articles.append(article[0])
			labels.append(article[1])

	print("Vectorizing ngrams...")
	vectorizer = CountVectorizer(ngram_range=(lower_bound, upper_bound), stop_words="english", min_df=calc_df(
		upper_bound, len(labels)))
	features = vectorizer.fit_transform(articles).toarray()

	if include_tfidf:
		print("Performing term-frequency times inverse document-frequency transformation...")
		transformer = TfidfTransformer(smooth_idf=False)
		features = transformer.fit_transform(features).toarray()

	print("Dumping tokenized features...")
	if not os.path.exists("data"):
		os.mkdir("data")
	with open("data/" + filename + ".json", "w") as f:
		json.dump(features.tolist(), f)
	with open("data/labels.json", "w") as f:
		json.dump(labels, f)

if __name__ == "__main__":
	main(sys.argv[1:])

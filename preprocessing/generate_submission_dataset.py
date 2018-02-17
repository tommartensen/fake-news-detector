import csv
import json
import sys
from time import time

import json_lines
import os
import re
import random


def increase_field_limit():
	"""
	Increases the field limit for csv files.
	Source: https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
	:return:
	"""
	max_int = sys.maxsize
	decrement = True

	while decrement:
		decrement = False
		try:
			csv.field_size_limit(max_int)
		except OverflowError:
			max_int = int(max_int / 10)
			decrement = True


def read_fake_news():
	"""
	Reads fake news articles from file.
	:return: Array of fake news with label.
	"""
	print("Reading fake news articles...")
	test_data = []
	with open("raw_data/fake.csv", mode="r", encoding="utf-8") as f:
		for article in csv.DictReader(f):
			test_data.append([clean_text(article["text"]), 1])
	return random.sample(test_data, 301)


def generate_sample():
	"""
	Generates a sample of genuine articles.
	:return: Array of genuine articles with label.
	"""
	articles = read_articles()
	print("Generating sample...")
	sample = random.sample(articles, 1201)
	test_data = []
	for article in sample:
		test_data.append([article, 0])
	return test_data


def read_articles():
	"""
	Reads sample article file.
	:return: Array of news articles.
	"""
	print("Reading genuine articles...")
	articles = []
	with open("raw_data/sample-1M.jsonl", mode="rb") as f:
		for article in json_lines.reader(f):
			if article["media-type"] == "News":						# filters blog entries
				articles.append(clean_text(article["content"]))
	return articles


def clean_text(text):
	"""
	Cleans a string from upper, special and unicode characters (ignored by double ascii conversion).
	:param text: string to clean
	:return: cleaned string
	"""
	cleaned_text = text.lower()
	for char in ["\n", "\t", "\r"]:
		cleaned_text = cleaned_text.replace(char, " ")
	cleaned_text = cleaned_text.encode("ascii", "ignore").decode("ascii")
	return re.sub(r"[^\w\s]", "", cleaned_text)


def split_fake_base(test_data_set):
	"""
	Splits the test_data set into training, test and validation set.
	:param test_data_set: test_data set to split
	:return: three test_data sets according to specification
	"""
	test_data_set = random.sample(test_data_set, len(test_data_set))
	return test_data_set[0:900], test_data_set[900:1200], test_data_set[1200:1500]


def save_test_data_set(test_data_set):
	"""
	Saves an array.
	:param test_data_set: array to save
	:return:
	"""
	print("Saving files...")
	if not os.path.exists("test_data"):
		os.mkdir("test_data")
	training_set, validation_set, test_set = split_fake_base(test_data_set)

	json.dump(test_data_set, open("test_data/fake_base_complete.json", "w"))
	json.dump(training_set, open("test_data/training_set.json", "w"))
	json.dump(validation_set, open("test_data/validation_set.json", "w"))
	json.dump(test_set, open("test_data/test_set.json", "w"))


if __name__ == "__main__":
	if not (os.path.exists("raw_data/fake.csv") and os.path.exists("raw_data/sample-1M.jsonl")):
		sys.exit("Critical Error!\nPlease place the raw data in the corresponding directory first.\nSee the readme "
		         "for more information.")
	start = time()
	increase_field_limit()
	fake_news_articles = read_fake_news()
	genuine_articles = generate_sample()
	save_test_data_set(fake_news_articles + genuine_articles)
	print("Operation successfully finished after %2.f seconds." % (time() - start))


#!/usr/bin/env python

import sys
import re
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
import glob

import logging 
logging.basicConfig()


from sklearn.datasets import fetch_20newsgroups
#newsgroups_train = fetch_20newsgroups(subset='train')
from pprint import pprint
#pprint(list(newsgroups_train.target_names))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import numpy as np

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))


# You can now see many things that these features have overfit to:
# Almost every group is distinguished by whether headers such as NNTP-Posting-Host: and Distribution: appear more or less often.
# Another significant feature involves whether the sender is affiliated with a university, as indicated either by their headers or their signature.
# Other features match the names and e-mail addresses of particular people who were posting at the time.


def test_without_preprocessing(classifier, vectorizer, categories):
	#Multinomial Naive Bayes classifier, which is fast to train and achieves a decent F-score
	newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=categories)
	vectors_test = vectorizer.transform(newsgroups_test.data)
	clf = MultinomialNB(alpha=.01)
	clf.fit(vectors, newsgroups_train.target)
	pred = clf.predict(vectors_test)
	metrics.f1_score(newsgroups_test.target, pred, average='macro')

# Remove headers, footers, quotes

def test_with_preprocessing(classifier, vectorizer, categories):
	newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
	vectors_test = vectorizer.transform(newsgroups_test.data)
	pred = clf.predict(vectors_test)
	metrics.f1_score(pred, newsgroups_test.target, average='macro')


def main():
	
	#Labels or Categories
	categories = ['alt.atheism',
	 'comp.graphics',
	 'comp.os.ms-windows.misc',
	 'comp.sys.ibm.pc.hardware',
	 'comp.sys.mac.hardware',
	 'comp.windows.x',
	 'misc.forsale',
	 'rec.autos',
	 'rec.motorcycles',
	 'rec.sport.baseball',
	 'rec.sport.hockey',
	 'sci.crypt',
	 'sci.electronics',
	 'sci.med',
	 'sci.space',
	 'soc.religion.christian',
	 'talk.politics.guns',
	 'talk.politics.mideast',
	 'talk.politics.misc',
	 'talk.religion.misc']

	#get the dataset
	newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
	pprint(list(newsgroups_train.target_names))
	print '\n\n'

	newsgroups_train.filenames.shape
	newsgroups_train.target.shape
	newsgroups_train.target[:10]

	#Turn the text into vectors of numerical values suitable for statistical analysis.
	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(newsgroups_train.data)
	vectors.shape

	# #Multinomial Naive Bayes classifier, which is fast to train and achieves a decent F-score
	# newsgroups_test = fetch_20newsgroups(subset='test',
 #                                     categories=categories)
	# vectors_test = vectorizer.transform(newsgroups_test.data)
	# clf = MultinomialNB(alpha=.01)
	# clf.fit(vectors, newsgroups_train.target)
	# pred = clf.predict(vectors_test)
	# metrics.f1_score(newsgroups_test.target, pred, average='macro')

	test_without_preprocessing(clf, vectorizer, newsgroups_train.target_names)
	test_with_preprocessing(clf, vectorizer, newsgroups_train.target_names)

	show_top10(clf, vectorizer, newsgroups_train.target_names)


if __name__ == '__main__':
    main()




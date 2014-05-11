"""
Preprocess articles crawled from arXiv API for LDA
"""

import re
import numpy as np
import itertools
import sqlite3

import sklearn.feature_extraction.text as sklearn_fe_text
import nltk

stopwords  = set(open('data/stopwords_english.txt', 'ru').read().split(','))

def tokenizer(doc):
	"""
	Preprocess a document and return a list of words
	"""
	# Convert the text to lower case
	message = doc.lower()
	# Replace \n by space
	message = re.sub('\n', ' ', message)
	# Remove all math expressions between dollar signs
	message = re.sub('\$.*?\$', ' ', message)
	# Remove latex markups
	message = re.sub('\\\w+|\\\\', ' ', message)
	# Remove all ponctuation
	punctuation = '[!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
	message = re.sub(punctuation, ' ', message)
	# Remove numbers
	message = re.sub('\d+', '', message)
	# Split the message in words
	words = message.split()
	# Remove stopwordsf = open('voc.txt', 'w')
	words = [w for w in words if not w in stopwords]
	# Remove too short words
	words = [w for w in words if len(w) > 1]
	# Stem the words to their root
	stemmer = nltk.PorterStemmer()
	words = [stemmer.stem(w) for w in words]
	return words


class parser(sklearn_fe_text.CountVectorizer):
	"""
	Class used to parse arXiv documents
	"""

	def __init__(self, vocab):
		super(parser, self).__init__(
			analyzer=tokenizer,
			vocabulary=vocab)
	
	def parse_doc_list(self, docs):
		"""
		Parse a document into a list of word ids and a list of counts,
		or parse a set of documents into two lists of lists of word ids
		and counts to make it consistent with Blei and Hoffman 
		onlineldavb implementation.

		Arguments: 
		docs:  List of D documents. Each document must be represented as
		       a single string. (Word order is unimportant.) Any
		       words not in the vocabulary will be ignored.
		vocab: Dictionary mapping from words to integer ids.

		Returns a pair of lists of lists. 

		The first, wordids, says what vocabulary tokens are present in
		each document. wordids[i][j] gives the jth unique token present in
		document i. (Don't count on these tokens being in any particular
		order.)

		The second, wordcts, says how many times each vocabulary token is
		present. wordcts[i][j] is the number of times that the token given
		by wordids[i][j] appears in document i.
		"""
		if (type(docs).__name__ == 'str'):
			temp = list()
			temp.append(docs)
			docs = temp

		td_sparsemat = self.fit_transform(docs)

		wordids = list()
		wordcts = list()
		for doc_vec in td_sparsemat:
			ids = list()
			cts = list()
			cx = doc_vec.tocoo()
			for word,count in itertools.izip(cx.col, cx.data):
				ids.append(word)
				cts.append(count)
			wordids.append(ids)
			wordcts.append(cts)

		return ((wordids, wordcts))


def create_voc(min_tc, min_df, max_df, db_path, vocabulary_filename):
	"""
	Create vocabulary

	Arguments :
	min_tc	minimum term count
	max_df	maximum document frequency
	"""
		
	conn = sqlite3.connect(db_path)
	c = conn.cursor()

	vectorizer = sklearn_fe_text.CountVectorizer(
		analyzer=tokenizer,
		vocabulary=None)

	docs = []
	print "Query..."
	for title, abstract in c.execute("SELECT title, abstract FROM Articles").fetchall():
		docs.append(abstract)

	print "Fit transform..."
	td_sparsemat = vectorizer.fit_transform(docs)
	M,V = td_sparsemat.shape

	print "Filter vocabulary..."
	# Compute term counts and documents frequencies
	termCounts = np.array(td_sparsemat.sum(axis=0))[0]	
	td_sparsemat.data = np.ones(len(td_sparsemat.data), dtype=np.int)
	documentCounts = np.array(td_sparsemat.sum(axis=0))[0]

	# Choose words to filter
	voc_indices = np.ones(V)
	voc_indices[termCounts < min_tc] = 0
	voc_indices[documentCounts < min_df * M] = 0
	voc_indices[documentCounts > max_df * M] = 0

	# Filter vocabulary
	inverse_voc = {v:k for k, v in vectorizer.vocabulary_.items()}
	voc = [inverse_voc[e] for e,i in enumerate(voc_indices) if i == 1]
	voc.sort()
	f = open(vocabulary_filename, 'w')
	for word in voc:
	    f.write(word+u"\n")
	f.close()

	return td_sparsemat, vectorizer, voc


if __name__ == "__main__":
	db_path = '../data/arxiv.db'
	vocabulary_filename = 'new_voc.txt'
	min_tc = 10
	min_df = 0.0001
	max_df = 0.6

	create_voc(min_tc, min_df, max_df, db_path, vocabulary_filename)

	vocab = open(vocabulary_filename, 'r').read().rstrip('\n').split('\n')

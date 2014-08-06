
"""
Preprocess articles crawled from arXiv API
"""

import re
import numpy as np
import itertools
import sqlite3
import unicodedata

import sklearn.feature_extraction.text as sklearn_fe_text
import nltk

from .stop_words import ENGLISH_STOP_WORDS

stemmer = nltk.PorterStemmer()

def tokenize_doc(doc):
	"""
	Preprocess a document and return a list of words
	"""
	if type(doc) == unicode:
		# Normalize doc
		doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore')
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
	# Remove stopwords
	words = [w for w in words if not w in ENGLISH_STOP_WORDS]
	# Remove too short words
	words = [w for w in words if len(w) > 1]
	# Stem the words to their root
	words = [stemmer.stem(w) for w in words]
	return words

def search_tokenize_author_training(authors_string):
	authors_list = []
	authors = authors_string.split('|')
	for author in authors:
		# Retrieve all name parts
		name_parts = author.split(" ")
		for name in name_parts:
			# Normalize unicode string to remove accents
			name = name.lower()
			name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore')
			authors_list.append(name)
	return authors_list

def search_tokenize_author_production(authors_string):
	authors_list = []
	authors = authors_string.split(' ')
	for name in authors:
		# Normalize unicode string to remove accents
		name = name.lower()
		name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore')
		authors_list.append(name)
	return authors_list

def recommender_tokenize_author(authors_string):
	authors_list = []
	authors = authors_string.split('|')
	for author in authors:
		# Normalize unicode string to remove accents
		author = author.lower()
		author = unicodedata.normalize('NFKD', author).encode('ASCII', 'ignore')
		author = author.split(' ')
		first = author[0]
		last = author[-1]
		authors_list.append(last+'_'+first[0])
	return authors_list


class AuthorVectorizer(sklearn_fe_text.CountVectorizer):
	"""
	Class used to parse raw authors from arXiv articles
	"""
	def __init__(self, vocabulary):
		"""
		Initialization

		Arguments:
		vocabulary : array_like
			Indicate the authors used in the vectorizer
		"""
		super(AuthorVectorizer, self).__init__(
			tokenizer = recommender_tokenize_author,
			vocabulary = vocabulary)

class SearchVectorizer(sklearn_fe_text.CountVectorizer):
	"""
	Class used for the search engine to parse authors or titles 
	of arXiv articles
	"""

	def __init__(self, category, training=False, vocabulary=None):
		"""
		Initialization

		Arguments:
		category : string ('author' or 'title')
			Indicate the type of vectorizer for tokenization
		vocabulary : array_like
			Indicate the word used for title description or the author names
			(Mandatory if training is False)
		"""
		assert category in ['author','title'], \
			"Invalid category: choose between 'author' and 'title'"
		
		if category == 'author':
			if training:
				super(SearchVectorizer, self).__init__(
					tokenizer = search_tokenize_author_training)
				self.vocabulary_ = {}
			else:
				super(SearchVectorizer, self).__init__(
					tokenizer = search_tokenize_author_production,
					vocabulary = vocabulary)
		if category == 'title':
			if training:
				super(SearchVectorizer, self).__init__(
					tokenizer = tokenize_doc)
			else:
				super(SearchVectorizer, self).__init__(
					tokenizer = tokenize_doc,
					vocabulary = vocabulary)



class ArticleParser(sklearn_fe_text.CountVectorizer):
	"""
	Class used to parse arXiv articles
	"""

	def __init__(self, vocab):
		"""
		Initialize an arXiv paper parser 

		Arguments:
		vocab : array_like
			Vocabulary
		"""
		super(ArticleParser, self).__init__(
			analyzer   = tokenize_doc,
			vocabulary = vocab)
	
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


def build_voc(min_tc, min_df, max_df, cursor):
	"""
	Create vocabulary

	Arguments :
	min_tc	minimum term count
	min_df  minimum document frequency
	max_df	maximum document frequency
	cursor  sqlite3 connection cursor
	"""
		
	vectorizer = sklearn_fe_text.CountVectorizer(
		analyzer=tokenize_doc,
		vocabulary=None)

	docs = []
	print "Query..."
	for title, abstract in cursor.execute("SELECT title, abstract FROM Articles").fetchall():
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

	return td_sparsemat, vectorizer, voc

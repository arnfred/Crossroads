import numpy as np
import scipy.sparse as scsp
import sqlite3
import cPickle
import tables

from sklearn.preprocessing import normalize
import sklearn.neighbors

import util
from util import mystdout
from onlineldavb.myonlineldavb import OnlineLDA
from arxiv.preprocess import ArticleParser, AuthorVectorizer
from arxiv.preprocess import recommender_tokenize_author

import pdb

class RecommendationMethodInterface(object):
	"""
	Interface for all recommendation methods.
	Every object deals directly with the arXiv database 
	and is linked to a hdf5 file opened by its parent
	"""
	def __init__(self, h5file, db_path):
		self.db_path = db_path
		self.h5file = h5file
		try:
			self.group = getattr(self.h5file.root.recommendation_methods, self.__class__.__name__)
		except AttributeError:
			self.group = self.h5file.create_group("/recommendation_methods", self.__class__.__name__, self.__class__.__name__)
		self.load_miscellaneous()

	def load_miscellaneous(self):
		"""
		Initialize miscellaneous data as object arguments 
		See init_miscellaneous function for more informations
		"""
		misc_data = self.h5file.root.miscellaneous.read()[0]
		self.D = misc_data['D']
		self.start_date = misc_data['start_date']
		self.end_date = misc_data['end_date']
		self.categories = set(misc_data['categories'].split('|'))
		self.query_condition = util.make_query_condition(self.start_date, self.end_date, self.categories)


	def train(self):
		"""
		Train the recommendation method with the data (i.e. build feature vectors)
		"""
		raise NotImplementedError( "train not implemented for %s" % self.__class__ )

	def build_nearest_neighbors(self):
		"""
		Compute the nearest neighbors for all articles from the feature vectors
		"""
		raise NotImplementedError( "build_nearest_neighbors not implemented for %s" % self.__class__ )

	def get_nearest_neighbors(self, paper_id, k):
		"""
		Query for the k nearest neighbors of article paper_id
		"""
		raise NotImplementedError( "get_nearest_neighbors not implemented for %s" % self.__class__ )

	# ====================================================================================================

	def close_db_connection(self):
		"""
		Close database connection
		"""
		self.conn.close()
		self.cursor = None

	def open_db_connection(self):
		"""
		Open database connection
		"""
		self.conn = sqlite3.connect(self.db_path)
		self.cursor = self.conn.cursor()

	# ====================================================================================================

	def load_all(self):
		"""
		Syntaxic sugar for everything we need
		"""
		for node in self.h5file.list_nodes(self.group):
			if type(node) is tables.group.Group:
				# If the node is a group, 
				# then it contains a sparse matrix and load it
				name = node._v_pathname.split('/')[-1]
				setattr(self, name, util.load_sparse_mat(name, self.h5file, node))
			else:
				# Else the node is an array/Carray, then load it
				self.load(self.group, node.name)

		# Build dictionary of indexes
		ids = self.h5file.root.main.ids[:]
		self.idx = dict(zip(ids[:], range(ids[:].shape[0])))

	def load(self, group, attr):
		"""
		Syntaxic sugar an attribute from hdf5 file, i.e.: 
		replace self.h5file.root.recommendation_methods.attr by self.attr to simplify code
		"""
		setattr(self, attr, getattr(group, attr))


# ======================================================================================================
# ======================================================================================================


class LDABasedRecommendation(RecommendationMethodInterface):

	def train(self, K, vocab_filename, batch_size=512, epochs_to_do=2):
		"""
		Train the recommender based on LDA

		Arguments:
		K : int
			Number of topics
		vocab_filename : string
			Location of vocabulary file
		batch_size : int (default: 512)
			Size of a batch of document per iteration of the algorithm
		epochs_to_do : int (default: 2)
			Number of epochs to do
		"""
		# Open db connection
		self.open_db_connection()
		# Initialize and save the Vocabulary
		self.vocabulary = np.array(open(vocab_filename, 'r').read().rstrip('\n').split('\n'))
		try:
			f = self.h5file.root.recommendation_methods.vocabulary
			f._f_remove()
			print "WARNING: array /recommendation_methods/vocabulary overwritten"
		except AttributeError:
			pass
		self.h5file.create_array(self.group, 'vocabulary', self.vocabulary, "Vocabulary")
		# Initialize the parser
		self.parser = ArticleParser(self.vocabulary)
		# Number of topics
		self.K = K
		# Vocabulary size
		self.W = len(self.parser.vocabulary_)
		# Initialize the online VB algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
		self.olda = OnlineLDA(self.parser, self.K, self.D, 1./self.K, 1./self.K, 1024., 0.7)
		# Actually run the algorithm and get the feature vector for each paper
		self._run_onlineldavb(batch_size, epochs_to_do)

	def _run_onlineldavb(self, batch_size, epochs_to_do):
		"""
		Run the online VB algorithm on the data
		"""
		# Initialize utility variables
		ids = list()           # A mapping between papers id and vector indices
		self.feature_vectors = [] 	# Feature vectors
		
		perplexity = 0.0
		self.perplexity = []

		# Run multiple over each documents
		for epoch in range(epochs_to_do):
			docs_seen = 0               # Number of documents seen up to now in this epoch
			iteration = 1               # Iteration in this epoch

			# Query for all documents we want
			query = self.cursor.execute(
				"""SELECT abstract,id FROM Articles WHERE %s ORDER BY updated_at""" % self.query_condition)

			# Run over the query results and feed them to the model per batch
			while True:
				# Fetch results
				result = query.fetchmany(batch_size)

				# Stop when there is no more documents
				if len(result) == 0:
					break

				cur_docset = [i for i,j in result] # Documents set
				docs_seen += len(result)

				if epoch < epochs_to_do-1:
					# Give them to online LDA
					(gamma, bound) = self.olda.update_lambda(cur_docset)

					# Compute an estimate of held-out perplexity
					(wordids, wordcts) = self.parser.parse_doc_list(cur_docset)
					perwordbound = bound * len(cur_docset) / (self.D * sum(map(sum, wordcts)))

					perplexity = np.exp(-perwordbound)
					if iteration%10 == 0:
						self.perplexity += [perplexity]

				if epoch == epochs_to_do-1:
					# In the last epoch, do not update the model (only get the feature vectors)
					gamma = self.olda.update_gamma(cur_docset)

					# Keep track of the feature vectors for each documment
					ids += [str(j) for i,j in result] # ids corresponding to current documents
					self.feature_vectors += self._compute_feature_vectors(gamma, addSmoothing=False).tolist()

				mystdout.write("Epoch %d: (%d/%d docs seen), perplexity = %e" % \
					(epoch,docs_seen,self.D,perplexity),
					iteration, np.ceil(float(self.D)/batch_size))
				iteration += 1

		mystdout.write("Online VB LDA done. perplexity = %.3f" % perplexity, 1,1, ln=1)

		# Convert the lists to a Numpy arrays
		self.feature_vectors = scsp.csr_matrix(self.feature_vectors, dtype=np.float)
		ids = np.array(ids)

		# Check indices coherence and reorder if necessary
		self.ids = self.h5file.root.main.ids
		if any(ids != self.ids[:]):
			print "Reorder articles indices..."
			idx = dict(zip(ids,range(len(ids))))
			ordering_func = np.vectorize(lambda x: idx[x])
			order = ordering_func(self.ids[:])
			self.feature_vectors = self.feature_vectors[order]
			ids = ids[order]
			assert all(ids == self.ids[:])

		print "Save feature_vectors to hdf5 file..."
		try:
			f = self.h5file.root.recommendation_methods.feature_vectors
			f._f_remove('recursive')
			print "WARNING: sparse array /recommendation_methods/feature_vectors are overwritten"
		except AttributeError:
			pass
		self.h5file.create_group(self.group, 'feature_vectors', 'Feature vectors sparse matrix')
		util.store_sparse_mat(self.feature_vectors, 'feature_vectors', self.h5file, self.group.feature_vectors)		

		print "Save topics to hdf5 file..."
		self.topics = self.olda._lambda
		try:
			f = self.h5file.root.recommendation_methods.topics
			f._f_remove()
			print "WARNING: array /recommendation_methods/topics overwritten"
		except AttributeError:
			pass
		self.h5file.create_array(self.group, 'topics', self.topics, "Topics")

	def _compute_feature_vectors(self, gamma, addSmoothing=False):
		"""
		Get the feature vectors from LDA gamma parameters
		"""
		if addSmoothing:
			# Usual feature vectors normalization
			return (gamma + self.olda._alpha) / (np.tile(gamma.sum(axis=1), \
				(gamma.shape[1],1)).T + gamma.shape[1]*self.olda._alpha)
		else:
			# Truncate vectors to remove minnimum values and get a sparse vector
			row_mins = np.min(gamma, axis=1)
			row_mins = np.tile(row_mins, (self.K,1)).T
			gamma[gamma == row_mins] = 0
			gamma /= np.tile(gamma.sum(axis=1), (gamma.shape[1],1)).T
			gamma[np.isnan(gamma)] = 1./self.K
			return gamma

	# ====================================================================================================

	def build_tree(self, metric):
		"""
		Build a ball tree to efficiently compare the the feature vectors
		"""
		self.btree = sklearn.neighbors.BallTree(self.feature_vectors,
			leaf_size=30, metric=metric)

	def load_tree(self, filename):
		"""
		Load the tree

		Arguments:
		filename : string
			Location of the file
		"""
		try:
			with open(filename, 'rb') as f:
				self.btree = cPickle.load(f)
		except IOError:
			print "File %s not found" % filename

	def save_tree(self, filename):
		"""
		Save the tree

		Arguments:
		filename : string
			Location of the file
		"""
		try:
			with open(filename, 'wb') as f:
				cPickle.dump(self.btree, f)
		except IOError:
			print "File %s not found" % filename

	# ====================================================================================================

	def build_nearest_neighbors(self, k=10, metric='manhattan'):
		"""
		Build the matrix a k nearest neighbors for every paper in the tree

		Argument:
		metric : string or callable (default: manhattan)
			A metric used to compare vectors, or a custom function
			Manhattan distance is used by default as it is a good metric to compare
			probability distributions
		"""

		try:
			self.btree
		except AttributeError:
			self.build_tree(metric)

		N,K = self.feature_vectors[:,:].shape
		batch_size = 100

		try:
			self.h5file.createCArray(self.group, "neighbors_distances", tables.Float64Atom(), shape=(N,k))
			self.h5file.createCArray(self.group, "neighbors_indices", tables.UInt64Atom(), shape=(N,k))
			self.neighbors_distances = self.group.neighbors_distances
			self.neighbors_indices = self.group.neighbors_indices
		except Exception:
			self.load("neighbors_distances")
			self.load("neighbors_indices")

		for i in range(N/batch_size):
			mystdout.write("Query knn... %d/%d"%(i*batch_size,N), i*batch_size,N)
			idx = range(i*batch_size, (i+1)*batch_size)
			distances, indices = self.btree.query(self.feature_vectors[idx,:], k=k+1)
			self.neighbors_distances[idx,:] = distances[:,1:]
			self.neighbors_indices[idx,:] = indices[:,1:]

		mystdout.write("Query knn... %d/%d"%(i*batch_size,N), i*batch_size,N, ln=1)

	# ====================================================================================================
	
	def get_nearest_neighbors(self, paper_id, k):
		"""
		Get the k nearest neighbors for a given paper_id

		Arguments:
		paper_id : int
			Id of the paper
		k : int
			Number of neighbors to return
		"""
		try:
			idx = self.idx[paper_id]
			distances = self.neighbors_distances[idx,:k]
			indices = self.neighbors_indices[idx,:k]
			return distances, indices
		except KeyError:
			print "Unknown paper id: %s" % paper_id

	def get_nearest_neighbors_online_2level(self, paper_id, k, percentile=1.0):
		"""
		Get the k nearest neighbors for a given paper_id by computing them online
		based on some metric
		"""
		M = self.feature_vectors.shape[0]
		try:
			self.others_jaccard
		except AttributeError:
			# Compute data to build the sparse matrix of top topics for all papers
			others_top = np.sum(np.cumsum(np.sort(self.feature_vectors[:], axis=1)[:,::-1], axis=1) \
				< percentile, axis=1)
			others = np.argsort(self.feature_vectors[:], axis=1)[:,::-1]
			others_col = []
			for i,t in enumerate(others_top):
				others_col += others[i,:t].tolist()
			others_row = np.repeat(np.arange(0,M), others_top)
			others_data = np.ones(others_top.sum())
			self.others_jaccard = scsp.coo_matrix(
							(
								others_data,
								(
									others_row,
									others_col
								),
							), shape=self.feature_vectors.shape, dtype=np.bool_)
			del others, others_top, others_data, others_row, others_col

		# Index of paper in feature vector matrix
		idx =self.idx[paper_id]
		# Jaccard feature vector for this paper, containing its top topics
		this_top = self.get_top_topics(idx, float(percentile))
		N = len(this_top)
		this_jaccard = scsp.coo_matrix(
			(
				np.ones(N),
				(
					this_top,                        
					np.zeros(N)
				),
			), shape=[self.feature_vectors.shape[1], 1], dtype=np.bool_)

		# Compute jaccard distances with all other papers
		jaccard_distances = np.dot(self.others_jaccard, this_jaccard).toarray().T[0]
		
		# Compute the euclidean distance for all the closest neighbors found via jaccard distance
		jaccard_indices = np.arange(M)[jaccard_distances == jaccard_distances.max()]
		distances = np.sum((self.feature_vectors[jaccard_indices,:] - np.tile(self.feature_vectors[idx], \
			(len(jaccard_indices),1)))**2, axis=1)
		indices = np.argsort(distances)[:k]
		distances = np.sort(distances)[:k]

		return distances, indices

	# ====================================================================================================

	def get_topic_top_words(self, topic_id, k):
		"""
		Get the top k words for a given topic (i.e. the ones with highest
		probability)

		Arguments:
		topic_id : int
			Id of the topic
		k : int
			Number of words to return
		"""
		return self.vocabulary[np.argsort(self.topics[topic_id][::-1])][:k]

	def get_top_topics(self, idx, k):
		"""
		Get the top k topics for a given paper (i.e. the ones with highest
		probability)

		Arguments:
		idx : int
			Id of the paper
		k : int or float
			if a int is provided: Number of topics to return
			if a float is provided: portion of topics (in probability) to return
		"""
		if type(k) is int:
			return np.argsort(self.feature_vectors[idx])[::-1][:k]
		elif type(k) is float:
			n = sum( np.cumsum(np.sort(self.feature_vectors[idx])[::-1]) < k)
			return np.argsort(self.feature_vectors[idx])[::-1][:n]


# ======================================================================================================
# ======================================================================================================


class AuthorBasedRecommendation(RecommendationMethodInterface):

	def train(self):
		print "Query documents..."
		sql_query_string = """SELECT id,authors
				FROM Articles
				WHERE %s
				ORDER BY updated_at""" % self.query_condition
		result = self.cursor.execute(sql_query_string).fetchall()
		ids = np.array([e[0] for e in result], dtype='S16')
		authors = np.array([e[1] for e in result])

		if any(ids != self.ids[:]):
			print "Reorder indices..."
			idx = dict(zip(ids,range(len(ids))))
			ordering_func = np.vectorize(lambda x: idx[x])
			order = ordering_func(self.ids[:])
			authors = authors[order]
			ids = ids[order]
			assert all(ids == self.ids[:])

		print "Build the set of authors..."
		self.author_vocabulary = map(recommender_tokenize_author, authors)
		self.author_vocabulary = [item for sublist in self.author_vocabulary for item in sublist]
		self.author_vocabulary = list(set(self.author_vocabulary))
		self.author_vocabulary = np.array(self.author_vocabulary)

		print "Transform data and build neighbors..."
		# Create the vectorizer
		author_vectorizer = AuthorVectorizer(vocabulary=self.author_vocabulary)
		# Vectorize the data
		tdmat = author_vectorizer.transform(authors)
		tdmat = tdmat.tocsr()
		tdmat = scsp.csr_matrix((tdmat.data, tdmat.indices, tdmat.indptr),shape=tdmat.shape,dtype='float')
		tdmat = normalize(tdmat, norm='l2', axis=1)
		self.author_neighbors = tdmat.dot(tdmat.T)

	def get_nearest_neighbors_authors(self, paper_id, k):
		# Index of paper in feature vector matrix
		idx = self.idx[paper_id]
		rec = self.author_neighbors[idx].toarray()[0]
		indices = np.argsort(rec)[::-1][:k]
		distances = np.sort(rec)[::-1][:k]
		distances = 0.3/distances

		return distances, indices


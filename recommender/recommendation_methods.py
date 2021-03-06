import numpy as np
import scipy.sparse as scsp
import sqlite3
import tables

import sklearn.metrics
from sklearn.preprocessing import normalize

import util
from util import mystdout
from onlineldavb.myonlineldavb import OnlineLDA
from arxiv.preprocess import ArticleParser, AuthorVectorizer
from arxiv.preprocess import recommender_tokenize_author

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
			# Initialize the hdf5 file group for this recommendation methods
			self.group = getattr(self.h5file.root.recommendation_methods, self.__class__.__name__)
		except AttributeError:
			# Create it if it does not exist
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

	def get_nearest_neighbors(self, paper_id, k=None):
		"""
		Get the k nearest neighbors for a given paper_id

		Arguments:
		paper_id : int
			Id of the paper
		k : int
			Number of neighbors to return. 
			If None, the function returns +inf for all articles not in neighbors_indices

		Returns:
		similarities : coo sparse matrix
			similarities of the neighbors
		indices : array (if k is not None)
			indices of the neighbors.
			If k is None, only similarities is returned, then the indices are the indices of 
			the vector similarities
		"""
		# Get the pre-computed indices and similarities of the top nearest neighbors stored
		# in the HDF5 file and reconstruct the vector of similarities for all documents.
		# Artcicles that are not present in the HDF5 file are assumed to have 0 similarity
		# with the current article. (They were actually very small and were threshold to 
		# zero to gain space and speed)
		try:
			idx = self.idx[paper_id]
			if k is None:
				# Get pre-computed values
				similarities = self.neighbors_similarities[idx,:]
				indices   = self.neighbors_indices[idx,:]
				# Reshape it
				similarities = scsp.coo_matrix(
					(
						similarities,
						(np.zeros(len(similarities)),indices)
					), shape=(1,self.D)).toarray()[0]
				return similarities
			else:
				similarities = self.neighbors_similarities[idx,:k]
				indices   = self.neighbors_indices[idx,:k]
				return similarities, indices
		except KeyError:
			print "Unknown paper id: %s" % paper_id
		except AttributeError:
			print "Neighbors not initialized in %s" % self.__class__.__name__

	def train(self):
		"""
		Train the recommendation method with the data (i.e. build feature vectors)
		"""
		raise NotImplementedError( "train not implemented for %s" % self.__class__.__name__ )

	def get_nearest_neighbors_online(self, paper_id, k):
		"""
		Get the k nearest neighbors for a given paper_id in online computations
		"""
		raise NotImplementedError( "build_nearest_neighbors not implemented for %s" % self.__class__.__name__ )

	def build_nearest_neighbors(self):
		"""
		Compute the nearest neighbors for all articles from the feature vectors
		"""
		raise NotImplementedError( "build_nearest_neighbors not implemented for %s" % self.__class__.__name__ )

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
		Syntaxic sugar for every array in the group.
		It creates a short to write ``self.attr'' instead of ``self.h5file.root.attr''
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

	def train(self, K, vocab_filename, n_titles=0, batch_size=512, epochs_to_do=2):
		"""
		Train the recommender based on LDA

		Arguments:
		K : int
			Number of topics
		w_title : int
			Weight of the title compared to the abstract (which is 1)
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
		# Weight of the titles
		self.w_title = w_title
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
				"""SELECT abstract,title,id FROM Articles WHERE %s ORDER BY updated_at""" % self.query_condition)

			# Run over the query results and feed them to the model per batch
			while True:
				# Fetch results
				result = query.fetchmany(batch_size)

				# Stop when there is no more documents
				if len(result) == 0:
					break

				cur_docset = [(e[1]+u' ')*self.w_title + e[0] for e in result] # Documents set
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
					ids += [str(e[2]) for e in result] # ids corresponding to current documents
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
			f = self.group.feature_vectors
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

	def build_nearest_neighbors(self, metric='total-variation', k=50):
		"""
		Build the matrix of nearest neighbors for every paper

		!!! WARNING !!!
		Only l2 metric works with sparse feature vectors

		Argument:
		metric : string or callable (default: euclidean)
			A metric used to compare vectors, or a custom function
			Manhattan distance is used by default as it is a good metric to compare
			probability distributions
		"""

		assert metric in ['total-variation', 'l1'], \
			"Invalid distance metric: choose between ['total-variation', 'l1']]"

		N = float(self.feature_vectors.shape[0])
		batch_size = 50

		try:
			f = self.group.neighbors_similarities
			f._f_remove()
			f = self.group.neighbors_indices
			f._f_remove()
		except AttributeError:
			pass
		self.h5file.create_carray(self.group, "neighbors_similarities", tables.Float64Atom(), shape=(N,k))
		self.h5file.create_carray(self.group, "neighbors_indices", tables.UInt64Atom(), shape=(N,k))
		self.neighbors_similarities = self.group.neighbors_similarities
		self.neighbors_indices = self.group.neighbors_indices

		for i in np.arange(np.ceil(N/batch_size)):
			mystdout.write("Query nearest neighbors... %d/%d"%(i*batch_size,N), i*batch_size,N)
			idx = np.arange(i*batch_size, (i+1)*batch_size)
			
			# If total-variation norm, then it is bounded by 1
			if metric is 'total-variation':
				dist = 0.5 * sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS['l1'](self.feature_vectors[idx], self.feature_vectors)
				self.neighbors_indices[idx,:] = np.argsort(dist, axis=1)[:,1:k+1]
				self.neighbors_similarities[idx,:] = 1 - np.sort(dist, axis=1)[:,1:k+1]
				self.h5file.flush()
			
			# If l-1 norm, then it is bounded by 2
			else:
				dist = sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS[metric](self.feature_vectors[idx], self.feature_vectors)
				self.neighbors_indices[idx,:] = np.argsort(dist, axis=1)[:,1:k+1]
				self.neighbors_similarities[idx,:] = 2 - np.sort(dist, axis=1)[:,1:k+1]
				self.h5file.flush()

		mystdout.write("Query nearest neighbors done.", i*batch_size,N, ln=1)

	# ====================================================================================================

	def get_nearest_neighbors_online_2level(self, paper_id, k, percentile=1.0):
		"""
		Get the k nearest neighbors for a given paper_id by computing them online
		based on some metric working in 2 levels: it first compute the Jaccard index
		to take as neighbors only the papers sharing the exact same top-k topics, and then
		class them using the euclidean distance
		"""

		print "WARNING: get_nearest_neighbors_online_2level function of LDABasedRecommendation " + \
				"is deprecated and might not work !"

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

	def get_nearest_neighbors_online(self, paper_id, metric='total-variation', kind='similarity'):
		"""
		Compute the k nearest neighbors for a given paper_id with online computations

		Arguments:
		paper_id : str
			id of the paper
		metric : str
			metric to choose nearest neighbors
		kind : str
			'distance' or 'similarity'
		"""
		assert kind in ['similarity', 'distance'], "Argument ``kind`` must be 'distance' or 'similarity'"

		if metric == 'total-variation':
			dist = 0.5 * np.array(abs(util.sparse_tile(self.feature_vectors[self.idx[paper_id]], (self.D,1)) - self.feature_vectors[:]).sum(axis=1).T)[0]
			if kind is 'distance':
				return dist
			else:
				return 1-dist
		elif metric in sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS.keys():
			dist = sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS[metric](self.feature_vectors[self.idx[paper_id]], self.feature_vectors[:])[0]
			if metric in ['l1','manhattan']:
				if kind is 'distance':
					return dist
				else:
					return 2-dist
			else:
				if kind is 'distance':
					return dist
				else:
					raise Exception("Argument ``kind`` cannot be 'similarity' with metric %s" % metric)
		else:
			raise Exception("Invalid metric")

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
		self.open_db_connection()
		result = self.cursor.execute("""SELECT id,authors FROM Articles 
			WHERE %s ORDER BY updated_at""" % self.query_condition).fetchall()
		ids = np.array([e[0] for e in result], dtype='S30')
		authors = np.array([e[1] for e in result])

		self.ids = self.h5file.root.main.ids
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

		print "Transform data and build feature vectors..."
		# Create the vectorizer
		author_vectorizer = AuthorVectorizer(vocabulary=self.author_vocabulary)
		# Vectorize the data
		tdmat = author_vectorizer.transform(authors)
		tdmat = tdmat.tocsr()
		tdmat = scsp.csr_matrix((tdmat.data, tdmat.indices, tdmat.indptr),shape=tdmat.shape,dtype='float')
		self.feature_vectors = normalize(tdmat, norm='l2', axis=1)
		del tdmat

		print "Save feature_vectors to hdf5 file..."
		try:
			f = self.group.feature_vectors
			f._f_remove('recursive')
			print "WARNING: sparse array /recommendation_methods/feature_vectors are overwritten"
		except AttributeError:
			pass
		self.h5file.create_group(self.group, 'feature_vectors', 'Feature vectors sparse matrix')
		util.store_sparse_mat(self.feature_vectors, 'feature_vectors', self.h5file, self.group.feature_vectors)
		self.h5file.flush()
		
	def build_nearest_neighbors(self, k=100):
		N = float(self.feature_vectors.shape[0])
		batch_size = 100

		try:
			f = self.group.neighbors_similarities
			f._f_remove()
			f = self.group.neighbors_indices
			f._f_remove()
		except AttributeError:
			pass
		self.h5file.create_carray(self.group, "neighbors_similarities", tables.Float64Atom(), shape=(N,k))
		self.h5file.create_carray(self.group, "neighbors_indices", tables.UInt64Atom(), shape=(N,k))
		self.neighbors_similarities = self.group.neighbors_similarities
		self.neighbors_indices = self.group.neighbors_indices
		
		all_sim = self.feature_vectors.dot(self.feature_vectors.T)

		for i in np.arange(np.ceil(N/batch_size)):
			mystdout.write("Query nearest neighbors... %d/%d"%(i*batch_size,N), i*batch_size,N)
			idx = np.arange(i*batch_size, (i+1)*batch_size)
			
			# Unsparse the cosine similarity, and convert it to a distance
			batch_sim = all_sim[idx,:].toarray()

			self.neighbors_indices[idx,:] = np.argsort(batch_sim, axis=1)[:,1:k+1]
			self.neighbors_similarities[idx,:] = np.sort(batch_sim, axis=1)[:,1:k+1]
			self.h5file.flush()

		mystdout.write("Query nearest neighbors done.", i*batch_size,N, ln=1)

	def get_nearest_neighbors_online(self, paper_id):
		# Compute cosine similarity
		sim = self.feature_vectors[self.idx[paper_id]].dot(self.feature_vectors.T)
		sim = sim.toarray()[0]
		return sim



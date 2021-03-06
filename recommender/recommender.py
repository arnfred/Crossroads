import numpy as np
import scipy.sparse as scsp
import sqlite3
import tables

import util
from util import mystdout
from .recommendation_methods import LDABasedRecommendation, AuthorBasedRecommendation
from .exceptions import UnknownIDException, UnknownAuthorException

END_DATE = '3000-01-01 00:00:00.000000'
START_DATE = '0001-01-01 00:00:00.000000'
CATEGORIES = set(['cs', 'math', 'q-bio', 'stat']) # See http://arxiv.org/help/api/user-manual for more information on arXiv categories

class ArXivRecommender():
	"""
	Content based recommender for arXiv papers based on LDA
	"""

	def __init__(self, hdf5_path, db_path, mode="r", start_date=START_DATE, end_date=END_DATE, categories=CATEGORIES):
		"""

		Arguments:
		hdf5_path : str
			Location of the hdf5 file. If it does not exist, it is created
		db_path : str
			Location of the db file
		mode : 'r','a','w'
			Opening mode of hdf5 file
			WARNING: if ``mode`` is 'w', HDF5 file gets overwritten at initialization
		start_date : string
			Starting date of the papers to process
		end_date : string
			Ending date of the papers to process
			Only papers in this range of date will be taken into account
		categories : iterable
			Categories of papers to process
		"""
		
		print "#  ArXivRecommender initialization:"

		# Path to Sqlite database
		self.db_path = db_path
		
		# Open hdf5 file
		self.h5file = tables.open_file(hdf5_path, mode=mode, title="Trailhead - arXiv recommender")
		print "#  HDF5 file '%s' opened in '%s' mode" % (self.h5file.filename, self.h5file.mode)

		# Initialize stuff
		if mode is 'w':
			self.init_miscellaneous(start_date, end_date, categories)
			self.init_main_group()
			self.init_recommendation_methods()
		else:
			# Load misc data
			self.load_miscellaneous()
			# Load main group data
			self.load_main_group()
			# Load recommendation methods
			self.load_recommendation_methods()

		# Get articles indices in recommender
		self.idx = dict(zip(self.ids,range(self.D)))

	def init_miscellaneous(self, start_date, end_date, categories):
		"""
		Initialize miscellaneous data as object arguments, i.e.:
		- D 			  : the number of articles in the system
		- start_date 	  : the start date of 'updated_at' fields of articles
		- end_date 		  : the end date of 'updated_at' fields of articles
		- categories 	  : the categories of articles
		- query_condition : the WHERE query condition to retrieve all articles
		"""
		# Create/overwrite miscellaneous table
		try:
			f = getattr(self.h5file.root, 'miscellaneous')
			f._f_remove()
			print "WARNING: group /miscellaneous overwritten"
		except AttributeError:
			pass
		misc_table = self.h5file.create_table('/', 'miscellaneous', MiscellaneousData, 
			'Miscellaneous variables referring to articles to take into account')
		misc_data = misc_table.row

		assert len(start_date) == 26, "Invalid start date. Must respect format YYYY-MM-DD HH:mm:ss.ssssss"
		self.start_date = start_date

		assert len(end_date) == 26, "Invalid end date. Must respect format YYYY-MM-DD HH:mm:ss.ssssss"
		self.end_date = end_date

		categories_string = '|'.join(categories)
		assert len(categories_string) <= 255, "categories_string is too large"
		self.categories = categories
		
		self.query_condition = util.make_query_condition(self.start_date, self.end_date, self.categories)
		self.open_db_connection()
		self.D = self.cursor.execute("SELECT COUNT(*) FROM Articles WHERE %s"%self.query_condition).fetchone()[0]
		misc_data['D'] = self.D
		misc_data['start_date'] = self.start_date
		misc_data['end_date'] = self.end_date
		misc_data['categories'] = categories_string
		misc_data.append()
		misc_table.flush()
		assert self.h5file.root.miscellaneous.nrows == 1, "There should not be multiple rows in table /miscellaneous"

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

	def init_main_group(self):
		"""
		Initialize the main group
		"""
		# Create/overwrite main group
		try:
			g = getattr(self.h5file.root, 'main')
			g._g_remove('recursive')
			print "WARNING: group /main overwritten on hdf5 file"
		except AttributeError:
			pass
		self.h5file.create_group("/", 'main', 'Recommender main group')
	
		# Get articles ids
		self.open_db_connection()
		result = self.cursor.execute("SELECT id FROM Articles WHERE %s ORDER BY updated_at"%self.query_condition).fetchall()
		self.ids = [e[0] for e in result]
		self.ids = np.array(self.ids, dtype='S30')
		
		# Store them
		util.store_carray(self.ids, 'ids', self.h5file, '/main')
		self.h5file.flush()

	def load_main_group(self):
		"""
		Load the main group
		"""
		self.ids = np.array(self.h5file.root.main.ids[:], dtype='S30')

	def init_recommendation_methods(self):
		"""
		Initialize the recommendation methods
		"""
		# Create/overwrite recommendation_methods group
		# (erase everything if it already exists)
		try:
			g = getattr(self.h5file.root, 'recommendation_methods')
			g._g_remove('recursive')
			print "WARNING: group /recommendation_methods overwritten on hdf5 file"
		except AttributeError:
			pass
		self.h5file.create_group("/", 'recommendation_methods', 'Recommendation methods')
		self.methods = dict()	# Dict of recommendation methods

	def load_recommendation_methods(self):
		print "#  Load recommendation methods:"
		self.methods = dict()	# Dict of recommendation methods
		# Iterate over the recommendation_methods group to get all recommendation methods
		# (The recommendations methods corresponding to all these groups will be loaded)
		group = self.h5file.root.recommendation_methods
		for i,n in enumerate(self.h5file.list_nodes(group)):
			assert type(n) is tables.group.Group, "group /recommendation_methods should only contain groups"
			class_name = n._v_name
			obj = globals()[class_name](self.h5file, self.db_path)
			obj.load_all()
			self.methods[class_name] = obj
			print "#    %s method loaded" % class_name

	# ====================================================================================================

	def add_recommendation_method(self, recommendation_method):
		"""
		Add a recommendation methods to the recommender.
		The method is just initialized but not trained or loaded

		Arguments:
		recommendation_method : str
			Name of a recommendation method in: LDABasedRecommendation, AuthorBasedRecommendation 
		"""
		obj = globals()[recommendation_method](self.h5file, self.db_path)
		self.methods[recommendation_method] = obj

	# ====================================================================================================

	def get_nearest_neighbors(self, paper_id, k):
		"""
		Get the k nearest neighbors of an article (use precomputed data from the hdf5 file)

		Arguments:
		paper_id : str
			arXiv id of the article
		k : int
			number of nearest neighbors to retrieve
		"""
		n_methods = len(self.methods)
		methods_idx = dict(zip(self.methods.keys(),range(n_methods)))
		methods_sim = np.zeros((n_methods, self.D), dtype=np.float)

		weights = np.array([
			[1.0, 0.0],
			[0.0, 1.0],
			[0.5, 0.5]])

		# Get neighbors and similarity for all methods
		for name,method in self.methods.iteritems():
			methods_sim[methods_idx[name]] = method.get_nearest_neighbors(paper_id)

		# Get the actual similarity and indices of the neighbors
		similarity, indices, methods_sim = self._get_nearest_neighbors(methods_sim, weights, k)

		return similarity, indices, methods_sim, methods_idx

	def _get_nearest_neighbors(self, methods_sim, weights, k):
		"""
		Internal computations of nearest neighbors shared by online and offline functions
		"""

		# Get the similarity for each weight vector
		similarity = weights.dot(methods_sim)

		# Sort the first k nearest neighbors
		indices_sorted = np.argpartition(-similarity, k, axis=1)[:,:k]

		# Save each weight vector result separately
		methods_sim = dict(zip(self.methods.keys(), similarity))

		# Get union of indices for all weight vectors
		n_methods,_ = indices_sorted.shape
		indices_set = set(indices_sorted.reshape([1,n_methods*k])[0])
		indices = np.array(list(indices_set))

		# Get mean similarity for each of these weight vectors
		similarity = similarity[:,indices]
		similarity = similarity.mean(axis=0)

		# Rescale similarity for a better display
		similarity /= similarity.sum()
		similarity *= 4

		# Sort the final recommendations
		indices_sorted = np.argsort(similarity)[::-1]
		indices = indices[indices_sorted]
		similarity = similarity[indices_sorted]

		return similarity, indices, methods_sim

	def get_nearest_neighbors_online(self, paper_id, k):
		"""
		Get the k nearest neighbors of an articles with online computations
		"""
		n_methods = len(self.methods)
		methods_idx = dict(zip(self.methods.keys(),range(n_methods)))
		methods_sim = np.zeros([n_methods, self.D])

		weights = np.array([
			[1.0, 0.0],
			[0.0, 1.0],
			[0.5, 0.5]])

		# Get neighbors and similarity for all methods
		for name,method in self.methods.iteritems():
			methods_sim[methods_idx[name]] = method.get_nearest_neighbors_online(paper_id)

		# Get the actual similarity and indices of the neighbors
		similarity, indices, methods_sim = self._get_nearest_neighbors(methods_sim, weights, k)

		return similarity, indices, methods_sim, methods_idx


	# ====================================================================================================

	def get_title(self, paper_id):
		"""
		Return the title of the paper with paper_id
		"""
		self.open_db_connection()
		try:
			title = self.cursor.execute("SELECT title FROM Articles WHERE id == ?", (paper_id,)).fetchone()[0]
		except IndexError:
			raise UnknownIDException(paper_id)

	def get_data(self, paper_id):
		"""
		Return all the data concerning the paper with paper_id in a dictionary where keys are
		column names and values are the data
		"""
		self.open_db_connection()
		try:
			data = self.cursor.execute("SELECT * FROM Articles WHERE id == ?", (paper_id,)).fetchone()
			names = [row[0] for row in self.cursor.description]
			return dict(zip(names,data))
		except TypeError:
			raise UnknownIDException(paper_id)

	def get_papers_from_author(self, author):
		"""
		Get all the paper ids of articles from a given author
		"""
		self.open_db_connection()
		formatted_author = "%{0}%".format(author)
		ids = self.cursor.execute("SELECT id FROM Articles WHERE authors LIKE ?", (formatted_author,)).fetchall()
		if len(ids) == 0:
			raise UnknownAuthorException(author)
		return ids

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


# ========================================================================================================
# ========================================================================================================

class MiscellaneousData(tables.IsDescription):
	D			= tables.UInt32Col()	# Number of articles in the recommender
	start_date 	= tables.StringCol(26)	# Starting date of articles
	end_date 	= tables.StringCol(26)	# Ending date of articles
	categories 	= tables.StringCol(255)	# '|' separated categories

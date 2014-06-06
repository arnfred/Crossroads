import numpy as np
import scipy.sparse as scsp
import sqlite3
import cPickle
import tables

from sklearn.preprocessing import normalize
import sklearn.neighbors

from onlineldavb.myonlineldavb import OnlineLDA
from arxiv.preprocess import ArticleParser, AuthorVectorizer
from arxiv.preprocess import recommender_tokenize_author
import util
from util import mystdout

END_DATE = '3000-01-01 00:00:00.000000'
START_DATE = '0001-01-01 00:00:00.000000'
CATEGORIES = set()

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
		start_date : string
			Starting date of the papers to process
		end_date : string
			Ending date of the papers to process
			Only papers in this range of date will be taken into account
		categories : iterable
			Categories of papers to process
		"""
		# Open hdf5 file system and sqlite database
		self.h5file = tables.open_file(hdf5_path, mode=mode, title="Trailhead - arXiv recommender")
		self.db_path = db_path

		# Initialize miscalleneous data
		self.init_miscellaneous(start_date, end_date, categories)

	def init_miscellaneous(self, start_date, end_date, categories):
		"""
		Initialize miscalleneous data (i.e. read or write it depending on file mode)
		"""
		self.open_db_connection()
		
		if self.h5file.mode is not 'r':
			try:
				n = self.h5file.root.miscalleneous
				n._f_remove()
			except AttributeError:
				pass
			misc_table = self.h5file.create_table('/', 'miscalleneous', MiscellaneousData, 'Miscellaneous variables \
					referring to articles to take into account')
			misc_data = misc_table.row

			assert len(start_date) == 26, "Invalid start date. Must respect format YYYY-MM-DD HH:mm:ss.ssssss"
			self.start_date = start_date

			assert len(end_date) == 26, "Invalid end date. Must respect format YYYY-MM-DD HH:mm:ss.ssssss"
			self.end_date = end_date

			categories_string = '|'.join(categories)
			assert len(categories_string) <= 255, "categories_string is too large"
			self.categories = categories
			self.query_condition = util.make_query_condition(start_date, end_date, categories)
			self.D = self.cursor.execute("""SELECT COUNT(*) FROM Articles
				WHERE %s """ % \
				(self.query_condition)).fetchone()[0]

			row['D'] = self.D
			row['start_date'] = self.start_date
			row['end_date'] = self.end_date
			row['categories'] = categories_string

		else:
			misc_data = self.h5file.root.miscalleneous.read()[0]
			self.D = misc_data['D']
			self.start_date = misc_data['start_date']
			self.end_date = misc_data['end_date']
			self.categories = set(misc_data['categories'].split('|'))

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
		try:
			data = self.cursor.execute("SELECT * FROM Articles WHERE id == ?", (paper_id,)).fetchone()
			names = [row[0] for row in self.cursor.description]
			return dict(zip(names,data))
		except TypeError:
			raise UnknownIDException(paper_id)

	def get_papers_from_author(self, author):
		self.open_db_connection()
		formatted_author = "%{0}%".format(author)
		ids = self.cursor.execute("SELECT id FROM Articles WHERE authors LIKE ?", (formatted_author,)).fetchall()
		if len(ids) == 0:
			raise UnknownAuthorException(author)

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
		Make syntaxic sugar for everything we need
		"""
		# From the h5file
		group = self.h5file.root.recommender
		for array in self.h5file.list_nodes(group):
			self.load(array.name)
		# Build dictionary of indexes
		self.idx = dict(zip(self.ids[:], range(self.ids[:].shape[0])))

	def load(self, attr):
		"""
		Make syntaxic sugar an attribute from hdf5 file
		"""
		setattr(self, attr, getattr(self.h5file.root.recommender, attr))

	def save(self, attr):
		"""
		Save an attribute to the hdf5 file
		"""
		try:
			self.h5file.createArray(self.h5file.root.recommender, attr, getattr(self, attr))
		except tables.exceptions.NodeError:
			# If the array already exists, update it
			getattr(self.h5file.root, attr)[:] = getattr(self, attr)

	# ====================================================================================================

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
		k : int or fload
			if a int is provided: Number of topics to return
			if a float is provided: portion of topics (in probability) to return
		"""
		if type(k) is int:
			return np.argsort(self.feature_vectors[idx])[::-1][:k]
		elif type(k) is float:
			n = sum( np.cumsum(np.sort(self.feature_vectors[idx])[::-1]) < k)
			return np.argsort(self.feature_vectors[idx])[::-1][:n]

# ========================================================================================================

class UnknownIDException(Exception):
	"""
	Exception raised if a unknown paper id is queried
	"""
	def __init__(self, paper_id):
		self.paper_id = paper_id

	def __str__(self):
		return repr("UnknownIDException: Unknown paper id %s" % self.paper_id)

class UnknownAuthorException(Exception):
	"""
	Exception raised if a unknown paper id is queried
	"""
	def __init__(self, author):
		self.paper_id = author

	def __str__(self):
		return repr("UnknownAuthorException: Unknown author %s" % self.author)

# ========================================================================================================

class MiscellaneousData(tables.IsDescription):
	D			= tables.UInt32Col()	# Number of articles in the recommender
	start_date 	= tables.StringCol(26)	# Starting date of articles
	end_date 	= tables.StringCol(26)	# Ending date of articles
	categories 	= tables.StringCol(255)	# '|' separated categories

import numpy as np
import scipy.sparse as scsp
import sqlite3
import cPickle
import tables

from sklearn.preprocessing import normalize
import sklearn.neighbors

from .exceptions import UnknownIDException, UnknownAuthorException
from onlineldavb.myonlineldavb import OnlineLDA
from arxiv.preprocess import ArticleParser, AuthorVectorizer
from arxiv.preprocess import recommender_tokenize_author
import util
from util import mystdout

import pdb

END_DATE = '3000-01-01 00:00:00.000000'
START_DATE = '0001-01-01 00:00:00.000000'
CATEGORIES = set(['math', 'cs', 'q-bio'])

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
			Opening mode of hdf5 file (if it is not 'r', many arrays gets overwritten ad updated)
		start_date : string
			Starting date of the papers to process
		end_date : string
			Ending date of the papers to process
			Only papers in this range of date will be taken into account
		categories : iterable
			Categories of papers to process
		"""
		self.db_path = db_path
		# Open hdf5 file
		self.h5file = tables.open_file(hdf5_path, mode=mode, title="Trailhead - arXiv recommender")
		# Initialize stuff
		self.init_miscellaneous(start_date, end_date, categories)
		self.init_main_group()
		self.init_recommendation_methods()

	def init_miscellaneous(self, start_date, end_date, categories):
		"""
		Initialize miscellaneous data (i.e. read or write it depending on file mode)
		"""
		if self.h5file.mode is not 'r':
			# Create/overwrite miscellaneous table
			try:
				f = getattr(self.h5file.root, 'miscellaneous')
				f._f_remove()
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
			self.query_condition = util.make_query_condition(start_date, end_date, categories)
			self.open_db_connection()
			self.D = self.cursor.execute("SELECT COUNT(*) FROM Articles WHERE %s"%self.query_condition).fetchone()[0]

			misc_data['D'] = self.D
			misc_data['start_date'] = self.start_date
			misc_data['end_date'] = self.end_date
			misc_data['categories'] = categories_string
			misc_data.append()
			misc_table.flush()
			assert self.h5file.root.miscellaneous.nrows == 1, "There should not be multiple rows in table /miscellaneous"

		else:
			misc_data = self.h5file.root.miscellaneous.read()[0]
			self.D = misc_data['D']
			self.start_date = misc_data['start_date']
			self.end_date = misc_data['end_date']
			self.categories = set(misc_data['categories'].split('|'))
			self.query_condition = util.make_query_condition(self.start_date, self.end_date, self.categories)

	def init_main_group(self):
		"""
		Initialize the main group (ids/idx arrays)
		"""
		if self.h5file.mode is not 'r':
			# Create/overwrite main group
			try:
				g = getattr(self.h5file.root, 'main')
				g._g_remove('recursive')
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

		else:
			self.ids = np.array(self.h5file.root.main.ids[:], dtype='S30')

		# Get articles indices in recommender
		self.idx = dict(zip(self.ids,range(self.D)))

	def init_recommendation_methods(self):
		"""
		Initialize the recommendation methods
		"""
		self.methods = list()	# List of recommendation methods
		self.weights = list()	# List of weight for recommendation combination

	# ====================================================================================================

	def add_recommendation_method(self, recommendation_method):
		"""
		Add a recommendation methods to the recommender

		Arguments:
		recommendation_method : instance of an object  RecommendationMethodInterface
		"""
		self.methods.append(recommendation_method)
		self.weight.append(1)

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


# ========================================================================================================
# ========================================================================================================

class MiscellaneousData(tables.IsDescription):
	D			= tables.UInt32Col()	# Number of articles in the recommender
	start_date 	= tables.StringCol(26)	# Starting date of articles
	end_date 	= tables.StringCol(26)	# Ending date of articles
	categories 	= tables.StringCol(255)	# '|' separated categories

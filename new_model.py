import numpy as np
from scipy import sparse
import time

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util

from recommender.recommendation_methods import LDABasedRecommendation, AuthorBasedRecommendation

h5file_path = 'recommender/data/new_recommender.h5'
db_path = 'recommender/data/arxiv.db'

recommender = ArXivRecommender(h5file_path, db_path, mode='r',
	start_date = '2000-01-01 00:00:00.000000',
	end_date   = '2014-05-01 00:00:00.000000',
	categories = set(['cs', 'math', 'q-bio', 'stat']))

import numpy as np
from scipy import sparse
import time
import tables

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util
from recommender.util import mystdout

from recommender.recommendation_methods import LDABasedRecommendation, AuthorBasedRecommendation
from recommender import search



h5file_path = 'recommender/data/recommender.h5'
db_path = 'recommender/data/arxiv.db'
paper_id = '1402.1774'

recommender = ArXivRecommender(h5file_path, db_path, mode='a')

self = recommender.methods['LDABasedRecommendation']

self.build_nearest_neighbors(k=50)
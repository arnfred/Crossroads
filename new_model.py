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

from recommender import search

h5file_path = 'recommender/data/new_recommender.h5'
db_path = 'recommender/data/arxiv.db'

recommender = ArXivRecommender(h5file_path, db_path, mode='a')
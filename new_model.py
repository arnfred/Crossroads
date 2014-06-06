import numpy as np
from scipy import sparse
import time

from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util

from recommender.recommendation_technics import *

recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db', mode='r',
				start_date = '0001-01-01 00:00:00.000000', end_date = '3000-01-01 00:00:00.000000',
				categories = set(['math', 'cs']))

author_recommendation = LDABasedRecommendation(recommender.h5file, recommender.db_path)

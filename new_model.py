import numpy as np
from scipy import sparse
import time

from sklearn.preprocessing import normalize

import recommender
reload(recommender)
from recommender.recommender import ArXivRecommender
from recommender import util

from .recommendation_methods import LDABasedRecommendation, AuthorBasedRecommendation

recommender = ArXivRecommender('recommender/data/recommender.h5', 'recommender/data/arxiv.db', mode='a',
				start_date = '2000-01-01 00:00:00.000000', end_date = '2014-05-01 00:00:00.000000',
				categories = set(['cs', 'math', 'q-bio', 'stat']))